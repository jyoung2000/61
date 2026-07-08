/**
 * Project export / import helpers.
 *
 * Export is a plain authenticated GET that streams a `.clipai.zip`; because
 * auth is a session cookie the browser sends automatically, we can just point
 * an anchor at the endpoint and let the browser handle the (potentially
 * multi-GB) download. Import POSTs the chosen zip with an XHR so we get an
 * upload-progress callback.
 */

/**
 * Trigger a browser download of a project bundle.
 * @param {string} jobId
 * @param {object} [opts]
 * @param {boolean} [opts.includeCache=false]    bundle audio.wav/frames/demucs
 * @param {boolean} [opts.includeOutputs=false]  bundle rendered clips
 * @param {boolean} [opts.includeThumbnails=true]
 */
export function exportProject(jobId, opts = {}) {
  const params = new URLSearchParams({
    include_cache: String(opts.includeCache ?? false),
    include_outputs: String(opts.includeOutputs ?? false),
    include_thumbnails: String(opts.includeThumbnails ?? true),
  });
  const url = `/api/jobs/${encodeURIComponent(jobId)}/export?${params.toString()}`;

  // An anchor click keeps the download in the browser's own manager, so a
  // large file doesn't tie up a fetch() promise or buffer in JS memory.
  const a = document.createElement('a');
  a.href = url;
  a.rel = 'noopener';
  // The server sets Content-Disposition with the real filename; `download`
  // just signals intent (value is ignored when the header is present).
  a.download = '';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

/**
 * Export a project and RESOLVE only once the `.clipai.zip` has been saved to
 * the user's disk. Used by "export before delete" so we never delete a project
 * until its bundle is safely downloaded. Fetches to a Blob (buffered on the
 * user's machine, so run these one at a time for large videos), then saves it.
 * @param {string} jobId
 * @param {object} [opts] same flags as exportProject
 * @returns {Promise<void>}
 */
export async function exportProjectBlocking(jobId, opts = {}) {
  const params = new URLSearchParams({
    include_cache: String(opts.includeCache ?? false),
    include_outputs: String(opts.includeOutputs ?? false),
    include_thumbnails: String(opts.includeThumbnails ?? true),
  });
  const url = `/api/jobs/${encodeURIComponent(jobId)}/export?${params.toString()}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Export failed (HTTP ${resp.status})`);
  const dispo = resp.headers.get('content-disposition') || '';
  const m = /filename="?([^"]+)"?/i.exec(dispo);
  const name = (m && m[1]) || `${jobId}.clipai.zip`;
  const blob = await resp.blob();
  const objUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = objUrl;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objUrl);
}

/**
 * Export a project WITH a download-progress callback, then save the
 * `.clipai.zip` to disk. Uses an XHR so we get `onprogress` (the export
 * endpoint streams a FileResponse with Content-Length, so progress is
 * computable). The bundle is buffered as a Blob on the user's machine, so run
 * these one at a time for large videos. Resolves once the file has been saved.
 * @param {string} jobId
 * @param {object} [opts] same flags as exportProject
 * @param {(pct:number)=>void} [onProgress] 0–100 download progress (-1 = unknown size)
 * @returns {Promise<void>}
 */
export function exportProjectWithProgress(jobId, opts = {}, onProgress) {
  return new Promise((resolve, reject) => {
    const params = new URLSearchParams({
      include_cache: String(opts.includeCache ?? false),
      include_outputs: String(opts.includeOutputs ?? false),
      include_thumbnails: String(opts.includeThumbnails ?? true),
    });
    const url = `/api/jobs/${encodeURIComponent(jobId)}/export?${params.toString()}`;

    const xhr = new XMLHttpRequest();
    xhr.open('GET', url);
    xhr.responseType = 'blob';

    if (onProgress) {
      xhr.onprogress = (e) => {
        // lengthComputable is false when the server/proxy omits Content-Length
        // (e.g. chunked) — report -1 so the UI can show an indeterminate bar.
        if (e.lengthComputable && e.total > 0) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        } else {
          onProgress(-1);
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(`Export failed (HTTP ${xhr.status})`));
        return;
      }
      try {
        const dispo = xhr.getResponseHeader('content-disposition') || '';
        const m = /filename="?([^"]+)"?/i.exec(dispo);
        const name = (m && m[1]) || `${jobId}.clipai.zip`;
        const objUrl = URL.createObjectURL(xhr.response);
        const a = document.createElement('a');
        a.href = objUrl;
        a.download = name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(objUrl);
        if (onProgress) onProgress(100);
        resolve();
      } catch (err) {
        reject(err);
      }
    };
    xhr.onerror = () => reject(new Error('Network error during export'));
    xhr.send();
  });
}

/**
 * Upload a project bundle to create a new job.
 * @param {File} file  a `.clipai.zip` (or any .zip) chosen by the user
 * @param {(pct:number)=>void} [onProgress] 0–100 upload progress
 * @returns {Promise<{job_id:string, filename:string, status:string, clips_count:number}>}
 */
export function importProject(file, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append('file', file, file.name);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/jobs/import');

    if (onProgress && xhr.upload) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
      };
    }

    xhr.onload = () => {
      let body;
      try {
        body = JSON.parse(xhr.responseText || '{}');
      } catch {
        body = {};
      }
      if (xhr.status >= 200 && xhr.status < 300 && body.job_id) {
        resolve(body);
      } else {
        reject(new Error(body.detail || `Import failed (HTTP ${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new Error('Network error during import'));
    xhr.send(form);
  });
}
