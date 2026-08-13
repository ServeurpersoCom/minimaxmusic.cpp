// field descriptor table: single source of truth for MM3Request field knowledge.
//
// every field in MM3Request is listed here with its section and type.
// helpers derive clear and serialize logic from this table.
// adding a field = adding one line here.

import type { MM3Request } from './types.js';

export type FieldSection = 'content' | 'metadata' | 'lm' | 'flow' | 'post' | 'routing';

interface FieldDef {
	key: keyof MM3Request;
	section: FieldSection;
	type: 'str' | 'num';
}

export const FIELDS: readonly FieldDef[] = [
	// content: the prompt pair
	{ key: 'caption', section: 'content', type: 'str' },
	{ key: 'lyrics', section: 'content', type: 'str' },

	// metadata: song level settings
	{ key: 'duration', section: 'metadata', type: 'num' },

	// lm: autoregressive sampling settings
	{ key: 'lm_cfg', section: 'lm', type: 'num' },
	{ key: 'lm_top_k', section: 'lm', type: 'num' },
	{ key: 'lm_seed', section: 'lm', type: 'num' },

	// flow: flow matching settings
	{ key: 'steps', section: 'flow', type: 'num' },
	{ key: 'dit_cfg', section: 'flow', type: 'num' },
	{ key: 'seed', section: 'flow', type: 'num' },

	// post: output encoding settings
	{ key: 'peak_clip', section: 'post', type: 'num' },
	{ key: 'mp3_bitrate', section: 'post', type: 'num' },

	// routing: model selection, preserved across resets of other sections
	{ key: 'lm_model', section: 'routing', type: 'str' },
	{ key: 'depth_model', section: 'routing', type: 'str' },
	{ key: 'cond_model', section: 'routing', type: 'str' },
	{ key: 'dit_model', section: 'routing', type: 'str' },
	{ key: 'vae_model', section: 'routing', type: 'str' }
];

// convert to number, undefined if empty/NaN
export function num(v: unknown): number | undefined {
	if (v == null || v === '') return undefined;
	const n = Number(v);
	return isNaN(n) ? undefined : n;
}

// typed dynamic access helpers
type Rec = Record<string, unknown>;
function get(r: MM3Request, key: keyof MM3Request): unknown {
	return (r as unknown as Rec)[key];
}
function set(r: MM3Request, key: keyof MM3Request, val: unknown): void {
	(r as unknown as Rec)[key] = val;
}

// resolve a field value to its serialized form, undefined if empty
function resolveField(f: FieldDef, raw: unknown): unknown {
	if (f.type === 'num') {
		const n = num(raw);
		return n != null ? n : undefined;
	}
	return raw ? String(raw) : undefined;
}

// serialize non-empty fields for the server payload and JSON export
export function buildSparse(r: MM3Request): MM3Request {
	const out: MM3Request = { caption: String(r.caption || '') };
	for (const f of FIELDS) {
		if (f.key === 'caption') continue;
		const val = resolveField(f, get(r, f.key));
		if (val !== undefined) set(out, f.key, val);
	}
	return out;
}

// reset all fields in a section to empty/undefined
export function clearSection(r: MM3Request, section: FieldSection): void {
	for (const f of FIELDS) {
		if (f.section !== section) continue;
		set(r, f.key, f.type === 'str' ? '' : undefined);
	}
}
