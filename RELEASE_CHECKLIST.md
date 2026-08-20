# Release & archiving checklist (fills manuscript field C2)

One-time setup, ~10 minutes total. GitHub username: YanjinLyu (already substituted throughout).

## 1. Publish the repository on GitHub (journal requirement: GitHub, not GitLab)
```bash
cd mixtriad
git init && git add -A && git commit -m "MixTriad v1.0.0"
git branch -M main
git remote add origin https://github.com/YanjinLyu/mixtriad.git
git push -u origin main
git tag v1.0.0 && git push origin v1.0.0
```
Then on GitHub: Releases -> "Draft a new release" -> choose tag v1.0.0 -> Publish.

## 2. Mint the Zenodo DOI (optional but recommended)
1. Log in at zenodo.org with your GitHub account.
2. GitHub -> Settings -> flip the switch next to `YanjinLyu/mixtriad` (the shipped
   `.zenodo.json` provides all metadata automatically).
3. Re-publish the v1.0.0 release (or create v1.0.1) -> Zenodo archives it and
   shows a DOI badge, e.g. 10.5281/zenodo.1234567.

## 3. Update three files with the final links (search for "zenodo")
- Manuscript Table 1, field C2:  https://github.com/YanjinLyu/mixtriad/tree/v1.1.2
  (add "archived: https://doi.org/10.5281/zenodo.XXXXXXX" if step 2 done)
- CITATION.cff -> repository-code
- README.md -> Citing section

## 4. On acceptance
SoftwareX forks the repository into github.com/ElsevierSoftwareX under the
manuscript number; no action needed from you.
