# Browser helper: pairing and verification

The included Chromium Manifest V3 helper is an **experimental read-only bridge**
from a signed-in ChatGPT web session to your local Chat Recall service. It does
not replace ChatGPT's own search and does not modify, archive, or delete chats.
JSON-file import in the local app is the stable first-run path.

## Pair the local service

1. Start Chat Recall and open **Browser helper** in the app header.
2. Open Chrome's `chrome://extensions` page, enable developer mode, and choose
   **Load unpacked**. Select this repository's `browser-extension` directory.
3. Open the helper popup. Copy its exact `chrome-extension://<id>` origin into
   the Chat Recall server's extension-origin allowlist and restart the server.
   See the main README for the current server flag. Arbitrary website origins
   must not be added. A token alone does not bypass server origin checks.
4. Pair the loopback endpoint (default `http://127.0.0.1:8765`) and the app's
   import token. The helper saves this pairing in this browser profile's local
   extension storage. Do not commit or share the token.
5. Keep a signed-in ChatGPT tab open. Click **Import / continue history**. Review
   the popup's status; it does not claim complete account coverage.

Manual invocations read at most three ordinary-list pages of twenty conversations,
then checkpoint their progress. To finish without repeated clicks, enable
**Automatically finish this history import**. Automatic invocations read one page,
pace detail requests, then wait at least a minute before the next batch. The job
stops and unchecks itself when the ordinary list is caught up. It does not keep
polling indefinitely. Uncheck it to pause; an in-flight page may finish.

Chrome, a signed-in ChatGPT tab and the local service must stay available. A
persisted one-shot alarm resumes each batch and is reconstructed when the
extension worker starts again. Sleep/browser suspension may delay an alarm;
it cannot wake a closed browser or guarantee a wall-clock completion time.

## Session source and refusals

The helper first reads the already-loaded session from ChatGPT's known
`client-bootstrap` JSON script, within the signed-in page execution. It checks
required fields, a 2 MiB UTF-8 input cap, and known JWT expiry; this is not token
signature verification. Missing, malformed or expired page data falls back to
the existing session endpoint. Tokens and raw bootstrap data are never saved in
extension state or sent to the local archive. The popup reports only a safe
session-source label.

Backend HTTP 401, 403 and challenges pause automatic mode for attention without
an alternate credential retry. Refresh/sign in through ChatGPT normally when
required. Schema and local import failures also need attention.

HTTP 429 retains completed-page progress and, when locally accepted, successful
details from the interrupted page. Its offset stays unchanged; saved versions
are skipped on retry. The job honors a valid Retry-After minimum, otherwise
uses exponential backoff starting at 60 seconds with positive jitter and a
30-minute fallback cap. A longer valid server delay is never shortened. The
popup shows state, attempt and local retry time; automatic mode resumes only
after that deadline. Manual/reset/pause/resume actions cannot bypass it. An
unrepresentable delay blocks further requests for attention. No account or IP
rotation is used.

A page-session source avoids an unnecessary session-endpoint request; it does
not override ChatGPT access controls or guarantee future endpoint stability.

## Account and workspace boundaries

The local account namespace is derived from a SHA-256 digest of the signed-in
ChatGPT user ID and configured workspace ID. ChatGPT access tokens remain within
the page execution and are never sent to Chat Recall. A user-supplied explicit
workspace ID becomes the `ChatGPT-Account-ID` header; leave it blank for the
default account context. Verify workspace coverage separately. Switching
accounts during a running scan stops the scan before importing its next page.

The helper's optional JSON upload uses a separate `file-import` namespace. Use
the local app's file-import account field when a particular account label is
needed. Importing data into different labels creates intentionally separate
archives; the helper does not infer that manual and live labels are identical.

## What is and is not proven

Offline tests validate endpoint restrictions, schema rejection, timestamp
normalization, pagination state, and checkpoint handling. They do not validate
ChatGPT's live endpoints, automatic account-wide coverage, native-Mac-app
changes, sign-in, archived chats, or project discovery. The adapter uses
undocumented web endpoints and may break when ChatGPT changes.

Every receipt labels project and archived-chat discovery and native-app sync as
**unverified**. An ordinary-list scan is not evidence of a full-account scan.
Failed pages do not advance the checkpoint. A partial import never removes a
previously imported message. No title-only or visited-page capture is presented
as exhaustive account history.

Before promoting this helper beyond experimental status, verify on an explicitly
chosen account:

- An older conversation never opened in the helper is imported.
- A new phrase typed in the native Mac ChatGPT app becomes searchable without
  opening that conversation in Chrome, within the claimed active-helper interval.
- Edited and regenerated branches are preserved, and project/archived samples
  reconcile against an account export.
- Sign-out, a failed list page, a failed detail, a local-server outage, and a
  repeated pagination cursor produce visible incomplete status and safe retry.

Run pure adapter tests with `node browser-extension/test_adapter.mjs`. No real
account, network request, or credentials are used by that command.
