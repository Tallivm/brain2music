from dataclasses import dataclass
import inspect
import functools
import threading
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging
from huggingface_hub import hf_hub_download
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from src.data.torch_utils import check_device, slerp
from src.external.prompt_weighting import get_weighted_text_embeddings
from src.constants import RIFFUSION_CHECKPOINT, SCHEDULER_OPTIONS

from typing import Union, Optional, Any, Callable

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass(frozen=True)
class PromptInput:
    """Parameters for one end of interpolation."""
    # Text prompt fed into a CLIP model
    prompt: str
    # Random seed for denoising
    seed: int
    # Negative prompt to avoid (optional)
    negative_prompt: Optional[str] = None
    # Denoising strength
    denoising: float = 0.75
    # Classifier-free guidance strength
    guidance: float = 7.0


@dataclass(frozen=True)
class InferenceInput:
    """
    Parameters for a single run of the riffusion model, interpolating between
    a start and end set of PromptInputs. This is the API required for a request
    to the model server.
    """
    # Start point of interpolation
    start: PromptInput
    # End point of interpolation
    end: PromptInput
    # Interpolation alpha [0, 1]. A value of 0 uses start fully, a value of 1 uses end fully.
    alpha: float
    # Number of inner loops of the diffusion model
    num_inference_steps: int = 50
    # Which seed image to use
    seed_image_id: str = "og_beat"
    # ID of mask image to use
    mask_image_id: Optional[str] = None


class RiffusionPipeline(DiffusionPipeline):
    """Diffusers pipeline for doing a controlled img2img interpolation for audio generation."""
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str,
        use_traced_unet: bool = True,
        channels_last: bool = False,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        local_files_only: bool = False,
        low_cpu_mem_usage: bool = False,
        cache_dir: Optional[str] = None,
    ):

        device = check_device(device)
        if device == 'cpu':
            dtype = torch.float32

        pipeline = RiffusionPipeline.from_pretrained(
            checkpoint,
            revision="main",
            torch_dtype=dtype,
            # Disable the NSFW filter, causes incorrect false positives
            # TODO(hayk): Disable the "you have passed a non-standard module" warning from this.
            safety_checker=lambda images, **kwargs: (images, False),
            low_cpu_mem_usage=low_cpu_mem_usage,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        ).to(device)

        if channels_last:
            pipeline.unet.to(memory_format=torch.channels_last)

        # Optionally load a traced unet
        if checkpoint == "riffusion/riffusion-model-v1" and use_traced_unet:
            traced_unet = cls.load_traced_unet(
                checkpoint=checkpoint,
                subfolder="unet_traced",
                filename="unet_traced.pt",
                in_channels=pipeline.unet.in_channels,
                dtype=dtype,
                device=device,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )

            if traced_unet is not None:
                pipeline.unet = traced_unet

        model = pipeline.to(device)
        return model

    @staticmethod
    def load_traced_unet(
        checkpoint: str,
        subfolder: str,
        filename: str,
        in_channels: int,
        dtype: torch.dtype,
        device: str = "cuda",
        local_files_only=False,
        cache_dir: Optional[str] = None,
    ) -> Optional[torch.nn.Module]:
        """
        Load a traced unet from the huggingface hub. This can improve performance.
        """
        if device == "cpu" or device.lower().startswith("mps"):
            print("WARNING: Traced UNet only available for CUDA, skipping")
            return None

        # Download and load the traced unet
        unet_file = hf_hub_download(
            checkpoint,
            subfolder=subfolder,
            filename=filename,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        unet_traced = torch.jit.load(unet_file)

        # Wrap it in a torch module
        class TracedUNet(torch.nn.Module):
            @dataclass
            class UNet2DConditionOutput:
                sample: torch.FloatTensor

            def __init__(self):
                super().__init__()
                self.in_channels = device
                self.device = device
                self.dtype = dtype

            def forward(self, latent_model_input, t, encoder_hidden_states):
                sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
                return self.UNet2DConditionOutput(sample=sample)

        return TracedUNet()

    @property
    def device(self) -> str:
        return str(self.vae.device)

    @functools.lru_cache()
    def embed_text(self, text) -> torch.FloatTensor:
        """
        Takes in text and turns it into text embeddings.
        """
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return embed

    @functools.lru_cache()
    def embed_text_weighted(self, text) -> torch.FloatTensor:
        """
        Get text embedding with weights.
        """
        return get_weighted_text_embeddings(
            pipe=self,
            prompt=text,
            uncond_prompt=None,
            max_embeddings_multiples=3,
            no_boseos_middle=False,
            skip_parsing=False,
            skip_weighting=False,
        )[0]

    @torch.no_grad()
    def riffuse(
            self,
            inputs: InferenceInput,
            init_image: Image.Image,
            mask_image: Optional[Image.Image] = None,
            use_reweighting: bool = True,
    ) -> Image.Image:
        """
        Runs inference using interpolation with both img2img and text conditioning.

        Args:
            inputs: Parameter dataclass
            init_image: Image used for conditioning
            mask_image: White pixels in the mask will be replaced by noise and therefore repainted,
                        while black pixels will be preserved. It will be converted to a single
                        channel (luminance) before use.
            use_reweighting: Use prompt reweighting
        """
        alpha = inputs.alpha
        start = inputs.start
        end = inputs.end

        guidance_scale = start.guidance * (1.0 - alpha) + end.guidance * alpha

        # TODO(hayk): Always generate the seed on CPU?
        if self.device.lower().startswith("mps"):
            generator_start = torch.Generator(device="cpu").manual_seed(start.seed)
            generator_end = torch.Generator(device="cpu").manual_seed(end.seed)
        else:
            generator_start = torch.Generator(device=self.device).manual_seed(start.seed)
            generator_end = torch.Generator(device=self.device).manual_seed(end.seed)

        # Text encodings
        if use_reweighting:
            embed_start = self.embed_text_weighted(start.prompt)
            embed_end = self.embed_text_weighted(end.prompt)
        else:
            embed_start = self.embed_text(start.prompt)
            embed_end = self.embed_text(end.prompt)

        text_embedding = embed_start + alpha * (embed_end - embed_start)

        # Image latents
        init_image_torch = preprocess_image(init_image).to(
            device=self.device, dtype=embed_start.dtype
        )
        init_latent_dist = self.vae.encode(init_image_torch).latent_dist
        # TODO(hayk): Probably this seed should just be 0 always? Make it 100% symmetric. The
        # result is so close no matter the seed that it doesn't really add variety.
        if self.device.lower().startswith("mps"):
            generator = torch.Generator(device="cpu").manual_seed(start.seed)
        else:
            generator = torch.Generator(device=self.device).manual_seed(start.seed)

        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents

        # Prepare mask latent
        mask: Optional[torch.Tensor] = None
        if mask_image:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            mask = preprocess_mask(mask_image, scale_factor=vae_scale_factor).to(
                device=self.device, dtype=embed_start.dtype
            )

        outputs = self.interpolate_img2img(
            text_embeddings=text_embedding,
            init_latents=init_latents,
            mask=mask,
            generator_a=generator_start,
            generator_b=generator_end,
            interpolate_alpha=alpha,
            strength_a=start.denoising,
            strength_b=end.denoising,
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=guidance_scale,
        )

        return outputs["images"][0]

    @torch.no_grad()
    def interpolate_img2img(
        self,
        text_embeddings: torch.Tensor,
        init_latents: torch.Tensor,
        generator_a: torch.Generator,
        generator_b: torch.Generator,
        interpolate_alpha: float,
        mask: Optional[torch.Tensor] = None,
        strength_a: float = 0.8,
        strength_b: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, list[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: Optional[float] = 0.0,
        output_type: Optional[str] = "pil",
        **kwargs
    ):

        batch_size = text_embeddings.shape[0]

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError("The length of `negative_prompt` should be equal to batch_size.")
            else:
                uncond_tokens = negative_prompt

            # max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat_interleave(
                batch_size * num_images_per_prompt, dim=0
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents_dtype = text_embeddings.dtype

        strength = (1 - interpolate_alpha) * strength_a + interpolate_alpha * strength_b

        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(
            [timesteps] * batch_size * num_images_per_prompt, device=self.device
        )

        # add noise to latents using the timesteps
        noise_a = torch.randn(
            init_latents.shape, generator=generator_a, device=self.device, dtype=latents_dtype
        )
        noise_b = torch.randn(
            init_latents.shape, generator=generator_b, device=self.device, dtype=latents_dtype
        )
        noise = slerp(interpolate_alpha, noise_a, noise_b)
        init_latents_orig = init_latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same args
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents.clone()

        t_start = max(num_inference_steps - init_timestep + offset, 0)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if mask is not None:
                init_latents_proper = self.scheduler.add_noise(
                    init_latents_orig, noise, torch.tensor([t])
                )
                # import ipdb; ipdb.set_trace()
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

        latents = 1.0 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return dict(images=image, latents=latents, nsfw_content_detected=False)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for the model.
    """
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)

    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np[None].transpose(0, 3, 1, 2)

    image_torch = torch.from_numpy(image_np)

    return 2.0 * image_torch - 1.0


def preprocess_mask(mask: Image.Image, scale_factor: int = 8) -> torch.Tensor:
    """
    Preprocess a mask for the model.
    """
    # Convert to grayscale
    mask = mask.convert("L")

    # Resize to integer multiple of 32
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))
    mask = mask.resize((w // scale_factor, h // scale_factor), resample=Image.NEAREST)

    # Convert to numpy array and rescale
    mask_np = np.array(mask).astype(np.float32) / 255.0

    # Tile and transpose
    mask_np = np.tile(mask_np, (4, 1, 1))
    mask_np = mask_np[None].transpose(0, 1, 2, 3)  # what does this step do?

    # Invert to repaint white and keep black
    mask_np = 1 - mask_np  # repaint white, keep black

    return torch.from_numpy(mask_np)


def pipeline_lock() -> threading.Lock:
    """Singleton lock used to prevent concurrent access to any model pipeline."""
    return threading.Lock()


def get_scheduler(scheduler: str, config: Any) -> Any:
    """
    Construct a denoising scheduler from a string.
    """
    if scheduler == "PNDMScheduler":
        from diffusers import PNDMScheduler

        return PNDMScheduler.from_config(config)
    elif scheduler == "DPMSolverMultistepScheduler":
        from diffusers import DPMSolverMultistepScheduler

        return DPMSolverMultistepScheduler.from_config(config)
    elif scheduler == "DDIMScheduler":
        from diffusers import DDIMScheduler

        return DDIMScheduler.from_config(config)
    elif scheduler == "LMSDiscreteScheduler":
        from diffusers import LMSDiscreteScheduler

        return LMSDiscreteScheduler.from_config(config)
    elif scheduler == "EulerDiscreteScheduler":
        from diffusers import EulerDiscreteScheduler

        return EulerDiscreteScheduler.from_config(config)
    elif scheduler == "EulerAncestralDiscreteScheduler":
        from diffusers import EulerAncestralDiscreteScheduler

        return EulerAncestralDiscreteScheduler.from_config(config)
    else:
        raise ValueError(f"Unknown scheduler {scheduler}")


def load_stable_diffusion_img2img_pipeline(
    checkpoint: str = RIFFUSION_CHECKPOINT,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    scheduler: str = SCHEDULER_OPTIONS[0],
) -> StableDiffusionImg2ImgPipeline:
    """
    Load the image to image pipeline.
    TODO(hayk): Merge this into RiffusionPipeline to just load one model.
    """
    if device == "cpu" or device.lower().startswith("mps"):
        print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
        dtype = torch.float32

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        checkpoint,
        revision="main",
        torch_dtype=dtype,
        safety_checker=lambda images, **kwargs: (images, False),
    ).to(device)
    pipeline.scheduler = get_scheduler(scheduler, config=pipeline.scheduler.config)
    return pipeline


def run_img2img(
    prompt: str,
    init_image: Image.Image,
    denoising_strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    negative_prompt: Optional[str] = None,
    checkpoint: str = RIFFUSION_CHECKPOINT,
    device: str = "cuda",
    scheduler: str = SCHEDULER_OPTIONS[0],
    progress_callback: Optional[Callable[[float], Any]] = None,
) -> Image.Image:
    with pipeline_lock():
        pipeline = load_stable_diffusion_img2img_pipeline(
            checkpoint=checkpoint,
            device=device,
            scheduler=scheduler,
        )

        generator_device = "cpu" if device.lower().startswith("mps") else device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        num_expected_steps = max(int(num_inference_steps * denoising_strength), 1)

        def callback(step: int, tensor: torch.Tensor, foo: Any) -> None:
            if progress_callback is not None:
                progress_callback(step / num_expected_steps)

        result = pipeline(
            prompt=prompt,
            image=init_image,
            strength=denoising_strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt or None,
            num_images_per_prompt=1,
            generator=generator,
            callback=callback,
            callback_steps=1,
        )

        return result.images[0]
