#!/usr/bin/env python3
"""Interactive CLI for Vim motions via ChatGPT 4.1."""
import argparse
import os
from openai import OpenAI


def ask_motion(client: OpenAI, prompt_template: str, model: str, description: str) -> str:
    """Send description to ChatGPT and return the Vim motion."""
    prompt = prompt_template.format(description=description)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=20,
        n=1,
    )
    return response.choices[0].message.content.strip()


def interactive_loop(client: OpenAI, prompt_template: str, model: str) -> None:
    """Start an interactive REPL for querying motions."""
    try:
        while True:
            query = input("Describe motion> ").strip()
            if not query or query.lower() in {"exit", "quit"}:
                break
            result = ask_motion(client, prompt_template, model, query)
            print("Motion:", result)
    except KeyboardInterrupt:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Query ChatGPT for Vim motions.")
    parser.add_argument("description", nargs="*", help="Task description. If omitted, start interactive shell.")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model to use (default: gpt-4.1)")
    parser.add_argument("--prompt-file", default="extended_prompt.txt", help="Path to extended prompt template")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key")
    args = parser.parse_args()

    if args.api_key is None:
        parser.error("OpenAI API key required via --api-key or OPENAI_API_KEY")

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    client = OpenAI(api_key=args.api_key)

    if args.description:
        description = " ".join(args.description)
        print(ask_motion(client, prompt_template, args.model, description))
    else:
        interactive_loop(client, prompt_template, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
