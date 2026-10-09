# Health Insurance Appeal Explorer

A Streamlit app built on 19,245 real California Independent Medical Review (IMR) cases (2001-2016, public data from the California Department of Managed Health Care).

- **Look up an existing case** by Reference ID: its details fill in automatically, with the actual decision, the model's estimate, and how similar cases turned out.
- **Enter a new client's case**: pick diagnosis, treatment, age, gender, case type and year, and get the model's estimated chance the denial is overturned, next to the historical rate for similar cases. Cases entered in a session can be downloaded as CSV.

## Important limits

- This is a **research and exploration tool, not a decision tool**. The model only sees broad labels (no medical evidence), so it is only modestly better than chance (about 0.69 AUC-ROC on cases it never saw).
- The reviewer's written `Findings` text is **never** used for prediction. It is written after the decision and gives the answer away.
- Nothing entered in the app is stored on the server. The session list resets on refresh.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit screens |
| `core.py` | Data loading, model training, prediction (no Streamlit code, easy to test) |
| `data/Independent_Medical_Reviews.csv.gz` | The IMR dataset (compressed) |
| `requirements.txt` | Python packages |

## Run on your own computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first start trains the model (a few seconds), then it is cached.

## Put it online for free (GitHub + Streamlit Community Cloud)

1. Create a free account at github.com and make a **new repository** (for example `imr-appeal-explorer`).
2. Upload everything in this folder to the repository (keep the `data` folder structure). Using the website: **Add file > Upload files**, drag the files in, then **Commit changes**.
3. Go to **share.streamlit.io** and sign in with your GitHub account.
4. Click **Create app**, choose your repository, branch `main`, main file path `app.py`, then **Deploy**.
5. After a minute or two you get a public web link you can share.

Free apps go to sleep when nobody has used them for a while; the first visitor after that waits a few seconds for it to wake up.

## Data source

California Department of Managed Health Care, Independent Medical Review determinations (public dataset).
