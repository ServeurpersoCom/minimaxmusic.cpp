<script lang="ts">
	import {
		Play,
		Square,
		Pencil,
		Download,
		Trash2,
		ChevronDown,
		Heart,
		Type,
		TriangleAlert
	} from '@lucide/svelte';
	import { app, setRequest } from '../lib/state.svelte.js';
	import { deleteSong, putSong } from '../lib/db.js';
	import type { Song } from '../lib/types.js';
	import Waveform from './Waveform.svelte';
	import Menu, { type MenuItem } from './Menu.svelte';
	import Dialog from './Dialog.svelte';

	let { song }: { song: Song } = $props();

	let playing = $state(false);
	let time = $state(0);
	let dur = $state(0);

	function toggle() {
		playing = !playing;
	}

	function load() {
		app.name = song.name;
		setRequest({ ...song.request });
	}

	function downloadAudio() {
		const url = URL.createObjectURL(song.audio);
		const a = document.createElement('a');
		a.href = url;
		const safe = song.name.replace(/[\\/:*?"<>|\x00-\x1f]/g, '') || 'song';
		const ext = song.format === 'mp3' ? 'mp3' : 'wav';
		a.download = `${safe}.${ext}`;
		a.click();
		URL.revokeObjectURL(url);
	}

	let confirmDeleteOpen = $state(false);
	let confirmDeleteNonFavOpen = $state(false);
	let renameOpen = $state(false);
	let renameValue = $state('');

	async function toggleFavorite() {
		if (song.id == null) return;
		song.favorite = !song.favorite;
		await putSong($state.snapshot(song));
	}

	function openRename() {
		renameValue = song.name;
		renameOpen = true;
	}

	async function doRename() {
		if (song.id == null) return;
		const v = renameValue.trim();
		if (!v || v === song.name) return;
		song.name = v;
		await putSong($state.snapshot(song));
	}

	async function doRemove() {
		if (song.id == null) return;
		await deleteSong(song.id);
		const idx = app.songs.findIndex((s) => s.id === song.id);
		if (idx >= 0) app.songs.splice(idx, 1);
	}

	// Deletes every non-favorite track in the list, regardless of which
	// card the menu was opened from. The current card is included in the
	// purge if it is not flagged favorite.
	async function doRemoveNonFavorites() {
		const victims = app.songs.filter((s) => !s.favorite);
		for (const s of victims) {
			if (s.id == null) continue;
			await deleteSong(s.id);
		}
		app.songs = app.songs.filter((s) => s.favorite);
	}

	// MM:SS:XX (hundredths) for current position
	function fmtPos(s: number): string {
		const m = Math.floor(s / 60);
		const sec = Math.floor(s % 60);
		const cs = Math.floor((s * 100) % 100);
		return (
			String(m).padStart(2, '0') +
			':' +
			String(sec).padStart(2, '0') +
			':' +
			String(cs).padStart(2, '0')
		);
	}

	// MM:SS for total duration
	function fmtDur(s: number): string {
		const m = Math.floor(s / 60);
		const sec = Math.floor(s % 60);
		return String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
	}

	// Single action menu: one entry per user intent. Order mirrors a natural
	// flow (tweak prompt -> rename -> grab audio -> destroy).
	// Destructive entries open a confirm dialog.
	const actionItems: MenuItem[] = $derived([
		{ icon: Pencil, label: 'Edit prompt', onSelect: load },
		{ icon: Type, label: 'Rename song', onSelect: openRename },
		{ icon: Download, label: 'Download audio', onSelect: downloadAudio },
		{ icon: Trash2, label: 'Delete this track', onSelect: () => (confirmDeleteOpen = true) },
		{
			icon: TriangleAlert,
			label: 'Delete non-favorites',
			onSelect: () => (confirmDeleteNonFavOpen = true)
		}
	]);
</script>

<div class="card">
	<div class="card-header">
		<button class="icon-btn" onclick={toggle} title={playing ? 'Stop' : 'Play'}>
			{#if playing}
				<Square size={14} />
			{:else}
				<Play size={14} />
			{/if}
		</button>
		<span class="card-name">{song.name}</span>
		<Menu items={actionItems}>
			{#snippet trigger()}<ChevronDown size={14} /> Menu{/snippet}
		</Menu>
		<button
			class="icon-btn"
			onclick={toggleFavorite}
			title={song.favorite ? 'Unfavorite' : 'Favorite'}
		>
			<Heart size={14} fill={song.favorite ? 'currentColor' : 'none'} />
		</button>
	</div>
	<Waveform {song} bind:playing bind:time bind:dur />
	<div class="card-footer">
		<span class="format-badge">{song.format.toUpperCase()}</span>
		<span class="timecode">{fmtPos(time)} / {fmtDur(dur)}</span>
	</div>
</div>

<Dialog bind:open={confirmDeleteOpen} title="Delete this track?" onConfirm={doRemove} />

<Dialog
	bind:open={confirmDeleteNonFavOpen}
	title="Delete non-favorites?"
	onConfirm={doRemoveNonFavorites}
/>

<Dialog bind:open={renameOpen} title="Rename song" onConfirm={doRename}>
	{#snippet body()}
		<input type="text" class="rename-input" bind:value={renameValue} />
	{/snippet}
</Dialog>

<style>
	.card {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		padding: 0.5rem;
		border: none;
		border-radius: 4px;
		background: var(--bg-card);
	}
	.card-header {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.card-footer {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.card-name {
		font-size: 0.8rem;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		flex: 1;
	}
	.format-badge {
		font-size: 0.6rem;
		font-family: monospace;
		padding: 0.05rem 0.3rem;
		border-radius: 2px;
		background: var(--fg);
		color: var(--bg);
		flex-shrink: 0;
	}
	.timecode {
		font-size: 0.7rem;
		font-family: monospace;
		color: var(--fg);
		white-space: nowrap;
		flex: 1;
	}
	.icon-btn {
		background: none;
		border: none;
		cursor: pointer;
		padding: 0.15rem;
		color: var(--fg);
		display: flex;
		align-items: center;
		gap: 0.2rem;
		font-size: 0.8rem;
	}
	.icon-btn:hover {
		color: var(--focus);
	}
	.rename-input {
		width: 100%;
		background: var(--bg-input);
		border: none;
		border-radius: 3px;
		padding: 0.25rem 0.4rem;
		color: var(--fg);
		font-size: 0.8rem;
	}
	.rename-input:focus {
		outline: 1px solid var(--focus);
	}
</style>
