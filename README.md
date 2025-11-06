--- a/README.md
+++ b/README.md
@@ -1,3 +1,28 @@
 # Persian-Incursion-API
 
-connecting mygpt with api
+This FastAPI service exposes authoritative Persian Incursion rules helpers so you can run a full "human vs MyGPT" play loop. The latest engine-backed execution flow is already wired up, so you do **not** need to change the server code—just follow the turn structure below from your integration.
+
+## Human ⇄ MyGPT turn loop
+
+1. **Capture the current game state.** Serialize the board in the canonical JSON format used by the API (see `docs` for privacy/terms and inspect your existing saves for the schema).
+2. **Send the state to `/actions/enumerate`.** Choose the `side_to_move` (`"Israel"` or `"Iran"`). The response contains legal actions plus a `nonce` you will reuse.
+3. **Build a plan:**
+   * Either call `/plan/suggest` to let the service propose a default plan, or
+   * Ask MyGPT to assemble a plan from the enumerated actions (each action is referenced by its `action_id`).
+4. **Validate the plan** by POSTing `{ "nonce": ..., "plan": ... }` to `/plan/validate`. MyGPT should iterate until the response reports `"ok": true`.
+5. **Execute the plan** by POSTing the same payload to `/plan/execute`. Because the engine now replays the exact action payloads, the server returns the updated game state plus an execution log.
+6. **Apply the result locally.** Use the returned `state` as the new canonical board. The human player can now review the engine log, take their impulse, and repeat the process (starting at step 1) for the next AI response.
+
+### Convenience endpoint
+
+If you simply want MyGPT to pick and resolve a reasonable impulse automatically, call `/turn/ai_move`. It enumerates, validates, and executes a short sequence for the side you specify and returns the resulting state in one step.
+
+## Testing
+
+Run the same smoke check used in development:
+
+```bash
+python -m compileall .
+```
+
+The command must complete without errors before deploying or regenerating the API client.

