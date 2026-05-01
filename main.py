import argparse

from generator import GLOBAL_DEFAULTS, MODELS, generate


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.2 Klein image generation (Apple Silicon)")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="flux2-klein-4b",
        help="Model to use. Choices: " + ", ".join(MODELS.keys()),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Realistic macro photograph of a hermit crab using a soda can as its shell, "
            "partially emerging from the can, captured with sharp detail and natural colors, "
            "on a sunlit beach with soft shadows and a shallow depth of field, with blurred "
            "ocean waves in the background. The can has the text `BFL Diffusers` on it and "
            "it has a color gradient that start with #FF5733 at the top and transitions to "
            "#33FF57 at the bottom."
        ),
    )
    parser.add_argument("--steps",    type=int,   default=None,  help="Number of inference steps (default: model default)")
    parser.add_argument("--guidance", type=float, default=None,  help="Guidance scale (default: model default)")
    parser.add_argument("--seed",     type=int,   default=GLOBAL_DEFAULTS["seed"], help="Random seed")
    parser.add_argument("--width",    type=int,   default=GLOBAL_DEFAULTS["width"],  help="Output image width in pixels (default: 1024)")
    parser.add_argument("--height",   type=int,   default=GLOBAL_DEFAULTS["height"], help="Output image height in pixels (default: 1024)")
    parser.add_argument("--image",    type=str,   nargs="+",     help="One or more input images for editing (local path or URL)")
    parser.add_argument("--output",   type=str,   default=None,  help="Output file path (default: <model>.png)")
    parser.add_argument(
        "--small-decoder",
        action="store_true",
        default=False,
        help=(
            "Use the distilled FLUX.2-small-decoder VAE for Klein models (~1.4x faster "
            "decode, ~1.4x lower VRAM, minimal quality loss). Ignored for non-Klein models."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_id = MODELS[args.model]["repo"]
    output_path = args.output or f"{args.model}.png"
    mode = "edit" if args.image else "generate"

    print(f"Model  : {args.model} ({repo_id})")
    print(f"Mode   : {mode}{f' ({len(args.image)} input image(s))' if args.image else ''}")
    if args.steps is not None:
        print(f"Steps  : {args.steps}  |  Guidance: {args.guidance}  |  Seed: {args.seed}  |  Size: {args.width}x{args.height}")
    else:
        print(f"Steps  : (model default)  |  Guidance: {args.guidance if args.guidance is not None else '(model default)'}  |  Seed: {args.seed}  |  Size: {args.width}x{args.height}")
    if args.small_decoder and MODELS[args.model].get("supports_small_decoder"):
        print("Decoder: small (FLUX.2-small-decoder)")
    print(f"Output : {output_path}")

    image = generate(
        model=args.model,
        prompt=args.prompt,
        images=args.image,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        width=args.width,
        height=args.height,
        use_small_decoder=args.small_decoder,
    )

    image.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
