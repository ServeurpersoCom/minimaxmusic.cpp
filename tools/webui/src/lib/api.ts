import type { MM3Request, MM3Props } from './types.js';
import { FETCH_TIMEOUT_MS, JOB_POLL_MS } from './config.js';

// shared: submit a request and return the job ID
async function submitJob(url: string, init: RequestInit): Promise<string> {
	const res = await fetch(url, init);
	if (!res.ok) {
		const err = await res.json().catch(() => ({ error: res.statusText }));
		throw new Error(`${res.status} ${err.error || res.statusText}`);
	}
	const data = await res.json();
	return data.id;
}

// POST /synth: submit the full pipeline request, returns job ID.
// format fills output_format so the UI selector drives the encoding.
export function synthSubmit(req: MM3Request, format: string): Promise<string> {
	return submitJob('synth', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ ...req, output_format: format })
	});
}

// GET /job?id=X: poll job status
export async function jobStatus(id: string): Promise<string> {
	const res = await fetch(`job?id=${encodeURIComponent(id)}`, {
		signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
	});
	if (!res.ok) throw new Error(`${res.status} Job not found`);
	const data = await res.json();
	return data.status;
}

// poll until done, throws on failure or cancel.
// no timeout: long jobs (several minutes of music) can take a while.
// the user cancels via the Cancel button if needed.
// retries on network errors (TypeError) and timeouts (DOMException).
// propagates HTTP errors (404 = job evicted, server restarted).
export async function pollJob(id: string): Promise<void> {
	for (;;) {
		try {
			const status = await jobStatus(id);
			if (status === 'done') return;
			if (status === 'failed') throw new Error('Generation failed');
			if (status === 'cancelled') throw new Error('Cancelled');
		} catch (e) {
			if (e instanceof TypeError || e instanceof DOMException) {
				// network down or timeout: retry next cycle
			} else {
				throw e;
			}
		}
		await new Promise((r) => setTimeout(r, JOB_POLL_MS));
	}
}

// GET /job?id=X&result=1: fetch the WAV result
// GET /job?id=X&result=1: fetch job result. Batch jobs reply
// multipart/mixed with one audio part per track in song-major order;
// single-track jobs reply with the raw audio body.
export async function jobResultBlobs(id: string): Promise<Blob[]> {
	const res = await fetch(`job?id=${encodeURIComponent(id)}&result=1`);
	if (!res.ok) throw new Error(`${res.status} Result not ready`);
	const ct = res.headers.get('Content-Type') || '';
	if (!ct.startsWith('multipart/')) {
		return [await res.blob()];
	}
	const match = ct.match(/boundary=([^\s;]+)/);
	if (!match) throw new Error('Missing boundary in multipart response');
	return parseMultipartParts(new Uint8Array(await res.arrayBuffer()), match[1]);
}

// Split a multipart/mixed body on its boundary and return one Blob per
// part, typed by the part's own Content-Type header.
function parseMultipartParts(buf: Uint8Array, boundary: string): Blob[] {
	const enc = new TextEncoder();
	const delim = enc.encode('--' + boundary);
	const dec = new TextDecoder();
	const results: Blob[] = [];

	// find all boundary positions
	const positions: number[] = [];
	for (let i = 0; i <= buf.length - delim.length; i++) {
		let ok = true;
		for (let j = 0; j < delim.length; j++) {
			if (buf[i + j] !== delim[j]) {
				ok = false;
				break;
			}
		}
		if (ok) positions.push(i);
	}

	for (let p = 0; p < positions.length - 1; p++) {
		const partStart = positions[p] + delim.length + 2;
		const partEnd = positions[p + 1] - 2;
		if (partStart >= partEnd) continue;

		// split headers from body at \r\n\r\n
		let splitAt = -1;
		for (let i = partStart; i < partEnd - 3; i++) {
			if (buf[i] === 13 && buf[i + 1] === 10 && buf[i + 2] === 13 && buf[i + 3] === 10) {
				splitAt = i;
				break;
			}
		}
		if (splitAt < 0) continue;

		// scan headers for Content-Type. Headers are CRLF-separated ASCII.
		const headerText = dec.decode(buf.slice(partStart, splitAt));
		let contentType = 'application/octet-stream';
		for (const line of headerText.split(/\r\n/)) {
			const m = line.match(/^Content-Type:\s*(.+)$/i);
			if (m) {
				contentType = m[1].trim();
				break;
			}
		}

		const body = buf.slice(splitAt + 4, partEnd);
		results.push(new Blob([body], { type: contentType }));
	}

	return results;
}

// POST /job?id=X&cancel=1: cancel a specific job
export async function cancelJob(id: string): Promise<void> {
	await fetch(`job?id=${encodeURIComponent(id)}&cancel=1`, { method: 'POST' });
}

// GET /props: server config (2s timeout)
export async function props(): Promise<MM3Props> {
	const res = await fetch('props', {
		signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
	});
	if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
	return res.json();
}
