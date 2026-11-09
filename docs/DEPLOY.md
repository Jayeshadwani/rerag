# Deploying NoteRecall on Render

Written for the owner, who does the deployment. Nothing here has been run on Render: the images were
built and measured with local Docker only. Read "Before you start" first.

## Before you start

1. **Memory.** The fp32 model did not fit a 512 MB limit, so the API serves an int8 quantized model
   (proposal 009, addendum 1). Locally, under a 512 MB limit, the API image started and answered
   through retrieval; see the README table and `results/deploy_measurements.json`. Render's accounting
   may differ from Docker's, so watch the first deploy's memory graph. The dev Recall@10 of the int8 model
   is a little lower than torch's (README), and it was not run on the test split.
2. **Free hours.** If both services stay on the free plan, they share 750 instance hours a month; when
   they are used up both are suspended until the month resets.
3. **Cold starts.** A free service sleeps when idle; the first request after that takes about a minute.
   The UI says "waking up" when that happens.
4. **The data token is a Secret File, not an environment variable.** `Dockerfile.api` reads it with
   `--mount=type=secret,id=DATA_REPO_TOKEN,required=true`, so the build fails at once if it is missing.
   The clone step has been seen to work on Render with this setup; the rest of the build was checked
   locally only. If the secret is missing the build stops at the `index` stage and nothing is leaked.
   The private repository must never be copied into the
   final image: `tests/unit/test_deploy_files.py` checks the Dockerfile for that.

## Steps

1. **Render account.** Sign up at render.com.
2. **Connect GitHub.** In Render: Account Settings, GitHub, connect, and give it access to the `rerag`
   repository.
3. **A read-only token for the notes.** On GitHub: Settings, Developer settings, Fine-grained personal
   access tokens, new token. Resource owner: you. Repository access: only `rerag-data`. Permissions:
   Contents, read-only. Set an expiry. Copy the token once.
4. **Two secrets of your own.**
   - `OPENAI_API_KEY`: your key (the budget cap in `config/default.toml` applies to the process, not to
     your OpenAI account; set a monthly limit in the OpenAI dashboard as well).
   - `UI_API_SECRET`: a random string shared by both services, for example the output of
     `openssl rand -hex 32`.
5. **Create the Blueprint.** Render dashboard, New, Blueprint, pick the `rerag` repository, branch with
   `render.yaml`. Render lists the two services and asks for every variable marked `sync: false`:
   - `noterecall-api`: `OPENAI_API_KEY`, `UI_API_SECRET`. Then, in the service's Environment page, add a
     **Secret File** named `DATA_REPO_TOKEN` whose content is the token from step 3 (the Blueprint cannot
     declare secret files).
   - `noterecall-ui`: `UI_API_SECRET` (same value as the API), and `API_URL`: leave it empty for now.
6. **Wait for the API build.** It clones the notes, runs `sync`, `quantize-onnx` and `build-index` (several minutes on
   CPU) and ends with the service answering `GET /health`. Open
   `https://<api-service>.onrender.com/health`: it should show `"status":"ok"` and the chunk count.
7. **Set `API_URL`.** On `noterecall-ui`, Environment: `API_URL` = the API service's URL
   (`https://<api-service>.onrender.com`, no trailing path). Save; the UI redeploys.
8. **Try it.** Open the UI's URL, ask a question, ask a follow-up ("and why?"), and open a source card.
   After 15 questions from one IP in a UTC day the page shows "Daily limit reached".

## Afterwards

- The notes change: re-deploy `noterecall-api` (the index is built at image build time).
- Rotate `DATA_REPO_TOKEN` or `UI_API_SECRET`: change the value on Render and re-deploy; for
  `UI_API_SECRET` change it on both services.
- The daily counters live in the API process memory: a restart or deploy resets them.
- The API returns the text of cited notes to anyone who can reach its URL. Keep the URL out of public
  places if that matters.
