import type { Song, MM3Request } from './types.js';

const DB_NAME = 'mm3-songs';
const DB_VERSION = 1;
const STORE = 'songs';

// Module-scoped singleton: one IDB connection for the whole page lifetime.
// Connection is lazy, retriable on error (null reset), and shared by every
// put/get/delete call below.
let dbPromise: Promise<IDBDatabase> | null = null;

function open(): Promise<IDBDatabase> {
	if (dbPromise) return dbPromise;
	dbPromise = new Promise((resolve, reject) => {
		const req = indexedDB.open(DB_NAME, DB_VERSION);
		req.onupgradeneeded = () => {
			const db = req.result;
			if (!db.objectStoreNames.contains(STORE)) {
				db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
			}
		};
		req.onsuccess = () => resolve(req.result);
		req.onerror = () => {
			dbPromise = null;
			reject(req.error);
		};
	});
	return dbPromise;
}

// wrap a single IDB transaction operation into a promise
function tx<T>(mode: IDBTransactionMode, fn: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
	return open().then(
		(db) =>
			new Promise((resolve, reject) => {
				const store = db.transaction(STORE, mode).objectStore(STORE);
				const req = fn(store);
				req.onsuccess = () => resolve(req.result);
				req.onerror = () => reject(req.error);
			})
	);
}

export function putSong(song: Song): Promise<number> {
	// IDBValidKey -> number (autoIncrement)
	return tx('readwrite', (s) => s.put(song)) as Promise<number>;
}

export function getAllSongs(): Promise<Song[]> {
	return tx('readonly', (s) => s.getAll());
}

export function deleteSong(id: number): Promise<void> {
	return tx('readwrite', (s) => s.delete(id)) as Promise<void>;
}

// pending job: saved before polling starts, cleared on completion,
// so a page reload resumes polling and lands the finished song.

export interface PendingJob {
	id: string;
	name: string;
	request: MM3Request;
}

const JOB_KEY = 'mm3-job-synth';

export function saveJob(job: PendingJob) {
	localStorage.setItem(JOB_KEY, JSON.stringify(job));
}

export function loadJob(): PendingJob | null {
	const raw = localStorage.getItem(JOB_KEY);
	if (!raw) return null;
	try {
		return JSON.parse(raw);
	} catch {
		return null;
	}
}

export function clearJob() {
	localStorage.removeItem(JOB_KEY);
}
