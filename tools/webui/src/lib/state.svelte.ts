import type { MM3Request, MM3Props, Song } from './types.js';
import { OUTPUT_FORMATS } from './config.js';

const STORAGE_KEY = 'mm3';

interface Saved {
	name: string;
	volume: number;
	format: string;
	dark: boolean;
	logsOpen: boolean;
	request: MM3Request;
}

function load(): Saved {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw) {
			const parsed = JSON.parse(raw);
			return {
				name: parsed.name || '',
				volume: parsed.volume ?? 0.5,
				format: OUTPUT_FORMATS.includes(parsed.format) ? parsed.format : 'mp3',
				dark: parsed.dark ?? true,
				logsOpen: parsed.logsOpen ?? true,
				request: parsed.request || { caption: '' }
			};
		}
	} catch {
		// corrupt or unavailable
	}
	return {
		name: '',
		volume: 0.5,
		format: 'mp3',
		dark: false,
		logsOpen: true,
		request: { caption: '' }
	};
}

const saved = load();

export const app = $state({
	name: saved.name,
	volume: saved.volume,
	format: saved.format,
	dark: saved.dark,
	logsOpen: saved.logsOpen,
	request: saved.request as MM3Request,
	songs: [] as Song[],
	props: null as MM3Props | null,
	toast: '' as string,
	toastOk: false
});

let toastTimer = 0;

export function toast(msg: string, ms = 4000, ok = false) {
	clearTimeout(toastTimer);
	app.toast = msg;
	app.toastOk = ok;
	toastTimer = setTimeout(() => {
		app.toast = '';
	}, ms) as unknown as number;
}

// overwrite app.request, preserving model routing fields unless the
// incoming request provides them (non-empty string).
export function setRequest(incoming: MM3Request) {
	if (!incoming.lm_model) incoming.lm_model = app.request.lm_model;
	if (!incoming.depth_model) incoming.depth_model = app.request.depth_model;
	if (!incoming.cond_model) incoming.cond_model = app.request.cond_model;
	if (!incoming.dit_model) incoming.dit_model = app.request.dit_model;
	if (!incoming.vae_model) incoming.vae_model = app.request.vae_model;
	app.request = incoming;
}

// persist on every change
$effect.root(() => {
	$effect(() => {
		const data: Saved = {
			name: app.name,
			volume: app.volume,
			format: app.format,
			dark: app.dark,
			logsOpen: app.logsOpen,
			request: app.request
		};
		localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
	});
});
