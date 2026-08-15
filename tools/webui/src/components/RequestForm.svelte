<script lang="ts">
	import { onMount } from 'svelte';
	import { parse as yamlParse, stringify as yamlStringify } from 'yaml';
	import { RotateCcw, Download, FolderOpen, X } from '@lucide/svelte';
	import { app, toast, setRequest } from '../lib/state.svelte.js';
	import { example } from '../lib/example.js';
	import { synthSubmit, pollJob, jobResultTracks, cancelJob } from '../lib/api.js';
	import { putSong, getAllSongs, saveJob, loadJob, clearJob } from '../lib/db.js';
	import { num, buildSparse, clearSection } from '../lib/fields.js';
	import type { MM3Request, Song } from '../lib/types.js';
	import type { PendingJob } from '../lib/db.js';
	import Dialog from './Dialog.svelte';
	import DialogButton from './DialogButton.svelte';

	let busy = $state(false);
	let fileInput: HTMLInputElement;
	let saveFormatOpen = $state(false);

	let d = $derived(app.props?.defaults);

	// resume a pending job after page reload, or land a fresh submission.
	// shared tail of both the onMount resume and the generate path.
	async function landJob(job: PendingJob) {
		await pollJob(job.id);
		const tracks = await jobResultTracks(job.id);
		clearJob();
		// batch jobs land one card per track in song-major order. The card
		// keeps the track's replay request: audio_codes + exact seed.
		const now = Date.now();
		for (let i = 0; i < tracks.length; i++) {
			const r = tracks[i].request;
			const song: Song = {
				name: tracks.length > 1 ? `${job.name} ${i}` : job.name,
				format: app.format,
				created: now + i,
				caption: r.caption || '',
				seed: r.seed ?? 0,
				duration: r.duration ?? 0,
				request: r,
				audio: tracks[i].audio
			};
			song.id = await putSong(song);
		}
		app.songs = (await getAllSongs()).reverse();
	}

	// on mount: resume polling for a pending job in localStorage.
	onMount(() => {
		const job = loadJob();
		if (job) {
			busy = true;
			landJob(job)
				.catch(() => {
					clearJob();
				})
				.finally(() => {
					busy = false;
				});
		}
	});

	function reset() {
		app.name = '';
		setRequest({ caption: '' });
	}

	function saveAs(format: 'json' | 'yaml') {
		const req = buildRequest();
		const text = format === 'json' ? JSON.stringify(req, null, 2) : yamlStringify(req);
		const mime = format === 'json' ? 'application/json' : 'application/x-yaml';
		const blob = new Blob([text], { type: mime });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		const safe = app.name.replace(/[\\/:*?"<>|\x00-\x1f]/g, '') || 'request';
		a.download = `${safe}.${format}`;
		a.click();
		URL.revokeObjectURL(url);
	}

	function importJson() {
		fileInput.click();
	}

	function onFileSelected(e: Event) {
		const input = e.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;
		// reset so the same file can be re-opened
		input.value = '';

		const ext = file.name.split('.').pop()?.toLowerCase() || '';

		// JSON and YAML share the same load path: parse, push the request into
		// the form, and use the file basename as app.name.
		const parsers: Record<string, (s: string) => MM3Request> = {
			json: JSON.parse,
			yml: yamlParse,
			yaml: yamlParse
		};
		const parse = parsers[ext];
		if (!parse) {
			toast('Unsupported file type: ' + ext);
			return;
		}
		file
			.text()
			.then((text) => {
				const parsed = parse(text) as any;

				// optional title field: lets a LLM authored YAML/JSON pre-fill
				// the song name on import. Stripped before setRequest.
				const importedName =
					typeof parsed?.title === 'string' && parsed.title.trim() ? parsed.title.trim() : '';
				delete parsed.title;
				setRequest(parsed as MM3Request);
				app.name = importedName || file.name.replace(/\.(json|ya?ml)$/i, '') || 'Imported';
			})
			.catch(() => {
				toast(`Invalid ${ext.toUpperCase()} file`);
			});
	}

	// snapshot app.request into a clean MM3Request with proper types.
	// bind:value guarantees app.request always matches the DOM.
	function buildRequest(): MM3Request {
		return buildSparse(app.request);
	}

	// Example: pick a random official demo prompt and fill the form
	function pickExample() {
		setRequest(example());
	}

	// Generate: submit the request, poll until done, land the song card.
	// The webui resolves the seeds so the stored request reproduces the song.
	async function generate() {
		busy = true;
		try {
			const req = buildRequest();
			if (!req.caption?.trim() || !req.lyrics?.trim()) {
				toast('Caption and lyrics are required');
				return;
			}
			const userSeed = num(req.seed);
			const userLmSeed = num(req.lm_seed);
			req.seed =
				userSeed != null && userSeed >= 0 ? userSeed : Math.floor(Math.random() * 0x100000000);
			req.lm_seed =
				userLmSeed != null && userLmSeed >= 0
					? userLmSeed
					: Math.floor(Math.random() * 0x100000000);

			const jobId = await synthSubmit(req, app.format);
			const job: PendingJob = { id: jobId, name: app.name || 'Untitled', request: req };
			saveJob(job);
			await landJob(job);
		} catch (e: unknown) {
			toast(e instanceof Error ? e.message : String(e));
		} finally {
			busy = false;
		}
	}

	// cancel the active pipeline job
	async function cancelPipeline() {
		try {
			const job = loadJob();
			if (job) await cancelJob(job.id);
		} catch {}
	}

	function clearLmConfiguration() {
		clearSection(app.request, 'lm');
	}

	function clearAdvancedLm() {
		clearSection(app.request, 'lm_advanced');
	}

	function clearFlowMatching() {
		clearSection(app.request, 'flow');
	}

	function clearPost() {
		clearSection(app.request, 'post');
	}

	function ph(v: unknown): string {
		return v != null ? String(v) : '';
	}
</script>

<form class="request-form" onsubmit={(e) => e.preventDefault()}>
	<input
		type="file"
		accept=".json,.yml,.yaml"
		bind:this={fileInput}
		onchange={onFileSelected}
		hidden
	/>
	<div class="toolbar">
		<button type="button" onclick={importJson} title="Open JSON/YAML prompt"
			><FolderOpen size={14} /> Open</button
		>
		<button
			type="button"
			onclick={() => (saveFormatOpen = true)}
			title="Save prompt as JSON or YAML"><Download size={14} /> Save</button
		>
		<button type="button" onclick={reset} title="Reset prompt"><RotateCcw size={14} /> Reset</button
		>
	</div>

	<details>
		<summary>Models</summary>
		<div class="details-body">
			<div class="model-row">
				<span class="model-label">LM</span>
				<select
					class="model-select"
					bind:value={app.request.lm_model}
					title="Global language model for the autoregressive stage. Scanned from --models directory at startup."
				>
					{#each app.props?.models.lm ?? [] as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>
			<div class="model-row">
				<span class="model-label">Depth</span>
				<select
					class="model-select"
					bind:value={app.request.depth_model}
					title="RVQ depth decoder for the acoustic codebooks. Scanned from --models directory at startup."
				>
					{#each app.props?.models.depth ?? [] as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>
			<div class="model-row">
				<span class="model-label">Cond</span>
				<select
					class="model-select"
					bind:value={app.request.cond_model}
					title="Condition encoder feeding the DiT. Scanned from --models directory at startup."
				>
					{#each app.props?.models.cond ?? [] as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>
			<div class="model-row">
				<span class="model-label">DiT</span>
				<select
					class="model-select"
					bind:value={app.request.dit_model}
					title="Diffusion Transformer for the flow matching stage. Scanned from --models directory at startup."
				>
					{#each app.props?.models.dit ?? [] as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>
			<div class="model-row">
				<span class="model-label">VAE</span>
				<select
					class="model-select"
					bind:value={app.request.vae_model}
					title="Flow VAE decoder for latent to audio conversion. Scanned from --models directory at startup."
				>
					{#each app.props?.models.vae ?? [] as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>
		</div>
	</details>

	<div class="section-title">Name</div>
	<input type="text" bind:value={app.name} placeholder="Untitled" />

	<div class="section-title">Caption</div>
	<textarea
		rows="8"
		placeholder="Upbeat pop rock with driving guitars, energetic male vocals..."
		bind:value={app.request.caption}
	></textarea>

	<div class="section-title">Lyrics</div>
	<textarea
		rows="8"
		placeholder="[verse]&#10;Write your lyrics here..."
		bind:value={app.request.lyrics}
	></textarea>

	<div class="section-title section-header">
		LM configuration
		<button
			type="button"
			class="clear-btn"
			title="Clear LM configuration"
			onclick={clearLmConfiguration}
			aria-label="Clear LM configuration"
		>
			<X size={20} />
		</button>
	</div>
	<div class="meta-grid">
		<label
			>Duration <input
				type="text"
				placeholder={ph(d?.duration)}
				bind:value={app.request.duration}
			/></label
		>
		<label
			>LM batch <input
				type="text"
				placeholder={ph(d?.lm_batch_size)}
				bind:value={app.request.lm_batch_size}
			/></label
		>
		<label
			>LM seed <input
				type="text"
				placeholder={ph(d?.lm_seed)}
				bind:value={app.request.lm_seed}
			/></label
		>
	</div>

	<details class="has-clear">
		<summary>Advanced LM</summary>
		<button
			type="button"
			class="clear-btn details-clear"
			title="Clear advanced LM"
			onclick={clearAdvancedLm}
			aria-label="Clear advanced LM"
		>
			<X size={20} />
		</button>
		<div class="details-body">
			<div class="meta-grid">
				<label
					>CFG scale <input
						type="text"
						placeholder={ph(d?.lm_cfg)}
						bind:value={app.request.lm_cfg}
					/></label
				>
				<label
					>Top K <input
						type="text"
						placeholder={ph(d?.lm_top_k)}
						bind:value={app.request.lm_top_k}
					/></label
				>
			</div>
		</div>
	</details>

	<details class="has-clear">
		<summary>Flow matching parameters</summary>
		<button
			type="button"
			class="clear-btn details-clear"
			title="Clear flow matching parameters"
			onclick={clearFlowMatching}
			aria-label="Clear flow matching parameters"
		>
			<X size={20} />
		</button>
		<div class="details-body">
			<div class="meta-grid">
				<label
					>Steps <input
						type="text"
						placeholder={ph(d?.steps)}
						bind:value={app.request.steps}
					/></label
				>
				<label
					>CFG scale <input
						type="text"
						placeholder={ph(d?.dit_cfg)}
						bind:value={app.request.dit_cfg}
					/></label
				>
				<label
					>Batch <input
						type="text"
						placeholder={ph(d?.synth_batch_size)}
						bind:value={app.request.synth_batch_size}
					/></label
				>
				<label
					>Seed <input type="text" placeholder={ph(d?.seed)} bind:value={app.request.seed} /></label
				>
			</div>
		</div>
	</details>

	<details class="has-clear">
		<summary>Advanced and post-processing</summary>
		<button
			type="button"
			class="clear-btn details-clear"
			title="Clear advanced and post-processing"
			onclick={clearPost}
			aria-label="Clear advanced and post-processing"
		>
			<X size={20} />
		</button>
		<div class="details-body">
			<div class="meta-grid">
				<label
					>Peak clip <input
						type="text"
						placeholder={ph(d?.peak_clip)}
						bind:value={app.request.peak_clip}
					/></label
				>
				<label
					>MP3 bitrate <input
						type="text"
						placeholder={ph(d?.mp3_bitrate)}
						bind:value={app.request.mp3_bitrate}
					/></label
				>
				<label
					>Format <select
						bind:value={app.format}
						title="Output audio format. WAV32 outputs raw IEEE float without normalization."
					>
						<option value="mp3">MP3</option>
						<option value="wav16">WAV16</option>
						<option value="wav24">WAV24</option>
						<option value="wav32">WAV32</option>
					</select></label
				>
			</div>
		</div>
	</details>

	<div class="action-row">
		<button
			type="button"
			disabled={busy}
			onclick={pickExample}
			title="Pick a random official demo prompt">Example</button
		>
		<button
			type="button"
			disabled={busy}
			onclick={generate}
			title="Run the full pipeline: LM, depth decoder, flow matching DiT, VAE">Generate</button
		>
		<button type="button" disabled={!busy} onclick={cancelPipeline} title="Cancel the active job"
			>Cancel</button
		>
	</div>
</form>

<Dialog bind:open={saveFormatOpen} title="Save format">
	{#snippet actions(close)}
		<DialogButton onclick={close}>Cancel</DialogButton>
		<DialogButton
			onclick={() => {
				saveAs('json');
				close();
			}}>JSON</DialogButton
		>
		<DialogButton
			onclick={() => {
				saveAs('yaml');
				close();
			}}>YAML</DialogButton
		>
	{/snippet}
</Dialog>

<style>
	.request-form {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
	.toolbar {
		display: flex;
		gap: 0.5rem;
	}
	.toolbar button {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.3rem;
	}
	label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.85rem;
		color: var(--fg-dim);
	}
	.section-title {
		font-size: 0.85rem;
		color: var(--fg);
		font-weight: 600;
		padding: 0.4rem 0 0;
	}
	.section-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	.has-clear {
		position: relative;
	}
	.details-clear {
		position: absolute;
		top: 0.4rem;
		right: 0;
	}
	.clear-btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: 0;
		border: none;
		background: transparent;
		color: var(--fg-dim);
		cursor: pointer;
		line-height: 0;
	}
	.clear-btn:hover {
		color: var(--fg);
	}
	textarea,
	input[type='text'],
	select {
		font-family: inherit;
		font-size: 0.9rem;
		padding: 0.4rem 0.5rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--bg-input);
		color: var(--fg);
		resize: vertical;
	}
	textarea:focus,
	input:focus {
		outline: 2px solid var(--focus);
		outline-offset: -1px;
	}
	.meta-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(8rem, 1fr));
		gap: 0.5rem;
	}
	details summary {
		cursor: pointer;
		font-size: 0.85rem;
		color: var(--fg);
		font-weight: 600;
		padding: 0.4rem 0;
	}
	details summary:hover {
		color: var(--fg);
	}
	.details-body {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		padding: 0.25rem 0 0.5rem;
	}
	.model-row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.model-label {
		font-size: 0.85rem;
		color: var(--fg-dim);
		flex-shrink: 0;
		min-width: 2.6rem;
	}
	.model-select {
		flex: 1;
		min-width: 0;
	}
	.action-row {
		display: flex;
		gap: 0.5rem;
	}
	.action-row button {
		flex: 1;
	}
	button {
		padding: 0.5rem 1rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--bg-btn);
		color: var(--fg);
		cursor: pointer;
		font-size: 0.85rem;
	}
	button:hover:not(:disabled) {
		background: var(--bg-btn-hover);
	}
	button:disabled {
		opacity: 0.4;
	}
</style>
