// all example prompts bundled at build time (Vite eager glob).
// Official MiniMax Music 3 demo prompts, extracted from the reference
// clone ../../music3-demo (source of minimax-ai.github.io/music3-demo).

import type { MM3Request } from './types.js';

const modules = import.meta.glob('../../example/*.json', { eager: true });

const examples: Record<string, any>[] = Object.values(modules).map((m: any) => m.default ?? m);

// pick a random example and return an MM3Request
export function example(): MM3Request {
	const ex = examples[Math.floor(Math.random() * examples.length)];
	return { caption: String(ex.caption), lyrics: String(ex.lyrics), duration: Number(ex.duration) };
}
