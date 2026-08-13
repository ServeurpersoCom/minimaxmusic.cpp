// mirrors MM3Request from request.h
// all fields optional except caption and lyrics: unset = server applies default
export interface MM3Request {
	caption: string;
	lyrics?: string;
	duration?: number;
	steps?: number;
	seed?: number;
	lm_seed?: number;
	lm_cfg?: number;
	lm_top_k?: number;
	dit_cfg?: number;
	peak_clip?: number;
	output_format?: string;
	mp3_bitrate?: number;
	lm_model?: string;
	depth_model?: string;
	cond_model?: string;
	dit_model?: string;
	vae_model?: string;
}

// GET /props response
export interface MM3Props {
	version: string;
	models: {
		lm: string[];
		depth: string[];
		cond: string[];
		dit: string[];
		vae: string[];
	};
	defaults: MM3Request;
}

// what we store in IndexedDB per song
export interface Song {
	id?: number;
	name: string;
	format: string;
	created: number;
	caption: string;
	seed: number;
	duration: number;
	request: MM3Request;
	audio: Blob;
	// user-marked favorite, persisted across reloads. Acts as a sticky
	// flag for the bulk "Delete non-favorites" action.
	favorite?: boolean;
	// 4096 normalized peaks [0..1] cached after the first decode, so F5
	// and re-mounts skip decodeAudioData entirely. Downsampled at draw
	// time to whatever canvas width is on screen.
	peaks?: Float32Array;
}
