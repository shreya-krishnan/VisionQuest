const DATABASE_NAME = "visionquest-workspace";
const DATABASE_VERSION = 1;
const STORE_NAME = "sessions";
const SESSION_KEY = "latest";

let databasePromise;
let writeQueue = Promise.resolve();

function openDatabase() {
  if (typeof indexedDB === "undefined") {
    return Promise.reject(new Error("IndexedDB is unavailable in this browser."));
  }
  if (!databasePromise) {
    databasePromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains(STORE_NAME)) {
          database.createObjectStore(STORE_NAME);
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("Could not open workspace storage."));
      request.onblocked = () => reject(new Error("Workspace storage is blocked by another tab."));
    }).catch((error) => {
      databasePromise = null;
      throw error;
    });
  }
  return databasePromise;
}

function transact(mode, operation) {
  return openDatabase().then((database) => new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, mode);
    const request = operation(transaction.objectStore(STORE_NAME));
    let value;
    request.onsuccess = () => { value = request.result; };
    request.onerror = () => reject(request.error || new Error("Workspace storage request failed."));
    transaction.oncomplete = () => resolve(value);
    transaction.onerror = () => reject(transaction.error || new Error("Workspace storage transaction failed."));
    transaction.onabort = () => reject(transaction.error || new Error("Workspace storage transaction was aborted."));
  }));
}

export function readWorkspaceSession() {
  return transact("readonly", (store) => store.get(SESSION_KEY));
}

export function saveWorkspaceSession(session) {
  const write = () => transact("readwrite", (store) => store.put(session, SESSION_KEY));
  writeQueue = writeQueue.catch(() => {}).then(write);
  return writeQueue;
}

export function clearWorkspaceSession() {
  const clear = () => transact("readwrite", (store) => store.delete(SESSION_KEY));
  writeQueue = writeQueue.catch(() => {}).then(clear);
  return writeQueue;
}
