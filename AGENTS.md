# Agent instructions

- Use uv for all local and remote Python commands and environments.
- Make and validate code changes locally, commit and push, then run on server.
- Server: `ssh -p 4618 gonghang@10.157.197.4`.
- Repository: `/home/gonghang/codes/BGC-Prophet`.
- Server HTTP/HTTPS proxy: `http://127.0.0.1:7892`.
- Use complete official training data from Zenodo record 17112180.
- Download datasets on server only.
- Use our raw NG GenBank and annotations; encode proteins afresh with ESM2-8M.
- Current task: detection model only, seeds 1301–1305, GPU 1.
- Preserve original model and training settings; document protocol changes.
- Never overwrite existing experiments or stop unrelated processes.
- Keep credentials, datasets, environments and model outputs out of Git.
- Keep experiment details in `ANALYSIS.md`, not this file.
- After editing Chinese documents: `autocorrect --fix FILE`.
- After editing Markdown: `rumdl fmt FILE`, then `rumdl check FILE`.
- In Git Bash, never redirect to `nul` or `NUL`.
