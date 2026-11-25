# PIT WALL

A Formula 1 race-strategy replay console with original constructor team colours
and full-colour driver portraits. Pick any race in the included archive, move lap
by lap, and watch the timing tower, charts, race feed, win probabilities and
undercut estimate update from the same data. A second view lists every World
Champion from 1950 to 2026 and links back to the replay.

The page is static vanilla HTML, CSS and JavaScript. A Python pipeline turns the
raw Ergast-style CSV archive into 18 compact JSON files. Circuit outlines are
cached at build time and served as compact SVG paths; driver portraits are local
full-colour 160x160 PNG files. The browser never loads the raw lap CSVs and makes
no runtime CDN calls for data or images.

## Run the page

The page fetches JSON, so it must be served over HTTP rather than opened from
`file://`:

```powershell
cd C:\Users\munaw\Downloads\archive
python -m http.server 8000
```

Open <http://localhost:8000/>.

No JavaScript build step or npm package is required to use the page itself.
Modern Chromium, Firefox and Safari are supported.

## Rebuild the data

The source files live in the project root. `build_data.py` reads:

`races.csv`, `lap_times.csv`, `pit_stops.csv`, `results.csv`, `drivers.csv`,
`constructors.csv`, `circuits.csv`, `status.csv`, `qualifying.csv`,
`driver_standings.csv`, `constructor_standings.csv`.

Requirements: Python 3.11 or newer, pandas, numpy and scipy for `build_data.py`;
`requests` and Pillow for `fetch_drivers.py`.

```powershell
python -m pip install pandas numpy scipy requests pillow
python fetch_drivers.py    # optional on later rebuilds; skips existing PNGs
python build_data.py
```

Run `fetch_drivers.py` once before the first `build_data.py` run so the generated
`has_img` flags match the files on disk. The two offline build-time sources are
OpenF1 (`/v1/drivers` headshots) and Wikipedia/Commons `prop=pageimages`; failed
or missing portraits simply leave `has_img: false` and the page uses a monogram.

`build_data.py` also caches circuit geometry under `raw_circuits/` on its first
run (bacinger/f1-circuits GeoJSON for latitude/longitude outlines and
MisterSix2295/f1-circuits-svg for era variants). Later rebuilds read the cache,
so no network access is needed. The rebuild takes a few minutes. It rewrites:

| File | Contents |
| --- | --- |
| `data/index.json` | Seasons and race list, circuit pit-loss summaries, all-race fallback wear, model summary, counts, default race, all-time `champions`, `constructorColors` for all 214 refs, `circuitLayouts` keyed by circuitId, `driverImages` count |
| `data/model.json` | Five model coefficients, training scaler, train/test counts, held-out metrics, ten calibration bins |
| `data/s2011.json` ... `data/s2026.json` | One file per season: drivers, laps, positions, pit stops, neutralised laps, per-race tyre fit and pit loss |
| `assets/drivers/{driverRef}.png` | 108 local full-colour 160x160 RGBA portraits (created by `fetch_drivers.py`, not by `build_data.py`) |

Only 2011+ races that have both `lap_times.csv` and `pit_stops.csv` rows are
included: **329 races**, **370,450 car-laps** and **8,658 green-flag pit-loss
values**. The 2021 Belgian Grand Prix is excluded because the archive has no pit
stops for it. The 2026 Las Vegas, Qatar and Abu Dhabi rounds are excluded because
the archive has no lap or pit-stop rows for them. The data folder is about 4.60
MiB.

### Data rules

- `\N` is read as missing. Missing lap times are not filled.
- A driver with lap times but no `results.csv` row is kept. The page labels the
  driver `no result row in source data`; the team comes from qualifying (by car
  number if the driver row is absent). The page never drops or crashes on this
  case. The 2026 Brazilian GP entry for Fittipaldi is the built-in example.
- Tyre compounds and weather are **not in the source data**. Stints are inferred
  from pit stops. Tyre age is laps since the last stop, or laps since the start
  if the driver has not stopped.
- Pit `milliseconds` is the time spent in the pit lane, not stationary time.
  Lap times already include the pit lane, so cumulative gaps are built from lap
  times only.
- Neutralised laps: for each lap after lap 1, take the field median lap time
  among cars with a time, excluding in/out laps; at least five cars are needed.
  A lap is neutralised when its field median exceeds `1.08 x` the 15th
  percentile of the race's lap medians. Neutralised periods are stored as lap
  lists and shaded in every time-based chart. The source has no flag type, so
  the page says "neutralised" rather than claiming safety car, VSC or red flag.
- Pit loss is `in-lap + out-lap - 2 x median(three clean laps before the stop)`.
  Only green stops are used and values are kept between 5 and 45 seconds.
  A circuit needs at least 15 stops before its own median is used; otherwise the
  undercut calculator uses the global median and says so.
- Tyre wear is an OLS regression of lap time on stint age and race lap with
  driver fixed effects, on clean laps only. A race fit is used only when
  `n >= 100`, the slope is between 0.005 and 0.20 s/lap, residual sd is at most
  2.0 s, and no more than 20% of laps are neutralised. If any gate fails, the
  page shows why and falls back visibly to the all-race median slope and sd.
- Champions: for each season in `races.csv`, the final round with standings is
  used. Driver position 1 gives the champion and position 2 the runner-up margin.
  Wins and podiums are counted from `results.csv`; the champion's team is the
  constructor with the most entries for that driver that season. Constructor
  standings begin in 1958, so earlier seasons store `constructor: null`. The
  2026 archive currently stops at round 20 of 23, so its standings round is the
  latest available rather than the scheduled finale. `has_replay` is true for
  2011-2026 and `default_race_id` points to that season's last replay race.
- Circuit layouts: latitude/longitude outlines are projected with a cosine
  longitude correction, uniformly scaled into the `0 0 220 130` viewBox, and
  simplified with Ramer-Douglas-Peucker to 80-110 points. Era-specific SVG
  layouts (for example Spain 2021-2022 vs 2023-2025 and Abu Dhabi 2009-2020 vs
  2021-2025) are stored in `eras` and selected by race year. All 36 replay
  circuits have geometry; a `fallback: true` entry would render
  `LAYOUT UNAVAILABLE` instead of a broken SVG.
- Portraits: a driver record only gets `has_img: true` when
  `assets/drivers/{driverRef}.png` exists at build time. The browser therefore
  never probes for a missing file. Every image is displayed in its original full
  colour and falls back to a three-letter monogram if it fails to decode.
  `fetch_drivers.py` only converts to RGBA; it never desaturates.
- Constructor colours: `build_data.py` maps all 214 `constructorRef` values in
  `constructors.csv` to official `#RRGGBB` team colours (with a stable palette
  fallback for obscure historical one-offs). The map is written to
  `index.constructorColors`, copied onto champion constructor objects, champion
  driver objects and replay driver metadata, then applied to the tower, bars and
  SVG traces by `index.html`. Engine-badged variants inherit their parent team's
  colour; pre-1958 champions remain `null`/`—`.

## What is on the page

**Header.** A view switcher (`REPLAY CONSOLE` | `CHAMPIONS`) synced with `#replay`
and `#champions`. In the replay view: season and race selects, lap scrubber,
`PREV` / `PLAY` / `NEXT`, 1x-8x replay speed, light/dark inversion, the current
lap count and a focused-driver badge with portrait. The light theme is an exact
inversion of the monochrome palette.

**Circuit layout (left).** A monochrome SVG of the current circuit with a
start/finish tick, plus circuit name, country, round/lap count and median pit
loss. The track uses `currentColor` at 1.75px, so it inverts with the theme. The
era-specific layout is chosen by race year.

**Timing tower (left).** Position, driver portrait plus three-letter code, gap
to leader, interval to the car ahead, last lap, tyre age and win probability.
Lapped cars show `+1 L` (or more). Click a row to focus that driver. The selected
focus is thick solid white in charts; the car directly ahead is thick dashed;
other cars are thin grey. Portraits with `has_img: false` show a monogram badge.

**Champions (`#champions`).** A searchable table of all 77 seasons (1950-2026)
with the driver champion, portrait, championship team, points, winning margin,
wins and podiums, and the constructor champion. Seasons from 2011 onward have an
`OPEN REPLAY` button that switches back to `#replay` and loads that season's
finale. Pre-1958 constructor cells show `—`. The table reflows into cards below
700 px without horizontal overflow.

**Race feed (left).** Lead changes, pit stops with pit-lane time, fastest laps,
retirements and the start/end of inferred neutralised periods. Newest events are
shown first.

**Centre charts.**

- *Lap chart*: on-track position, lap 1 to the current lap.
- *Gap to leader*: cumulative lap-time difference, with pit lane time included.
- *Pace*: clean laps for the focus car and the car ahead, plus a dotted OLS line
  for the focus car's current stint (lap time against tyre age).
- *Strategy timeline*: stint blocks with pit-stop ticks. Compounds are absent.
- *Win probability trace*: full-race probability history for every car.

Neutralised periods are shaded on all five time-based charts.

**Win probability (right).** One bar per running car at the current lap. The
focus bar is solid, the car ahead is a dashed outline; if the focus leads, P2 is
used as the threat. The list is ordered by the current running order.

**Undercut (right).** A six-parameter Monte Carlo estimate for the focus/car-ahead
pair. If the focus leads, the calculator switches to "threat" mode and evaluates
P2 against the leader. Results show the probability the chaser is ahead after
both stops, median margin, 10th-90th percentile range, break-even gap, mean tyre
gain, and a probability-versus-gap curve. `Reset to live` restores the current
pair and values.

**Evidence (below).**

- *Pit loss by circuit*: median and IQR for every circuit, with the 15-stop
  minimum marked. The current circuit is highlighted.
- *Tyre wear by round*: per-race OLS slopes for the selected season. Hollow bars
  failed a reliability gate; the all-race median is drawn as a reference.
- *Win model*: held-out log loss, Brier score and accuracy for the full model
  and three baselines, plus a ten-bin calibration plot.

**Footer.** Source counts, what is missing from the data, and model limits.

Below 860 px the layout becomes one column. At 400 px the page has no horizontal
overflow; all seven timing-tower columns remain visible.

### Controls

- `Left` / `Right`: previous / next lap.
- `Space`: play or pause.
- Click a timing-tower row: select the focus car.
- `Reset to live`: restore the undercut sliders to the live pair.
- The theme choice is stored in `localStorage`.

## Undercut method

For the focus/car-ahead pair, the live gap, circuit pit loss, race tyre slope and
ahead tyre age come from the fitted data. The sliders expose:

`gap`, `pit loss`, `tyre wear`, `ahead tyre age`, `k laps`, `traffic cost`.

The deterministic mean tyre gain is

```text
advantage = wear slope x ahead tyre age x k
```

Each of **6,000 seeded draws** then computes

```text
offset = chaser pit loss + traffic cost - ahead pit loss - advantage + lap noise
P(chaser ahead) = P(gap + offset < 0)
```

Both cars draw pit loss from a normal distribution with the circuit median and
`IQR / 1.349` as the standard deviation, clipped to 5-45 s. Lap noise has standard
deviation `race residual sd x sqrt(2k)`. The margin is positive when the chaser
is ahead. The break-even gap is the gap at which the probability crosses 50%.
The verdict is solid `BOX NOW` at 65% or more, outlined `MARGINAL` from 45% to
65%, and plain `STAY OUT` below 45%.

The estimate is explicitly labelled as an estimate. It has no tyre-compound
model, no warm-up curve, no safety-car model and no traffic simulation beyond the
slider. If the race or circuit wear fit is unreliable, the page falls back to the
all-race median and prints the reason.

## Win model

A conditional logit with a softmax over the running cars. One winner is observed
per race-lap group. The five features are the ones in `data/model.json`:

1. `ln(position)`
2. `ln(position) x laps_left / total`
3. `min(gap_to_leader, 300) / sqrt(laps_left + 1)`
4. `clip(pace_vs_leader, -3, 5) x sqrt(laps_left + 1) / 10`
5. `clip(stops_made - field_median, -2, 2)`

Gap is cumulative lap-time difference; recent pace uses up to five clean laps.
Features are standardised with training mean and standard deviation. The fit uses
L2 regularisation with lambda `1e-3` and L-BFGS. Train is 2011-2022; test is
2023-2026.

| Split | Races | Race-lap groups | Rows |
| --- | ---: | ---: | ---: |
| Train 2011-2022 | 239 | 14,358 | 269,294 |
| Test 2023-2026 | 90 | 5,434 | 101,156 |

Held-out metrics:

| Model | Log loss | Brier | Accuracy |
| --- | ---: | ---: | ---: |
| Full model | 0.693153 | 0.324303 | 78.5057% |
| Position only | 0.859645 | 0.393775 | 76.4998% |
| Leader wins | 3.929637 | 0.470003 | 76.4998% |
| Uniform | 2.918344 | 0.945573 | 5.4427% |

Log loss is the mean negative log probability of the actual winner. Brier is
computed per race-lap, summed over running cars, then averaged over groups.
Accuracy is the share of race-lap groups where the highest-probability car won.
The uniform accuracy is expected random-guess accuracy, `mean(1/n)`. The full
model beats the position-only baseline by 0.166492 log loss. Ten calibration bins
are fitted on held-out car-laps and plotted on the model card.

The model deliberately ignores future pit stops, so probabilities wobble during
pit cycles. It does not model retirements, weather, tyre compounds, penalties or
race control. It is a historical replay model, not a live race predictor.

## Verification

Serve the page first, then run the two checks in separate terminals.

Browser check:

```powershell
npm install playwright
npx playwright install chromium   # once, if the bundled browser is missing
node verify_page.js
```

If Playwright's bundled browser is not installed, point `PITWALL_CHROMIUM` at an
existing Chrome/Chromium executable:

```powershell
$env:PITWALL_CHROMIUM = "C:\path\to\chrome.exe"
```

The script writes `page_debug.json` and the final screenshots
`screenshot-dark.png`, `screenshot-light.png` and `screenshot-400.png`. Run it
before `verify.py`, which reads `page_debug.json`. It also fails if the page
makes any request outside the local server, so the no-CDN rule stays enforced.

Independent pandas recomputation:

```powershell
python verify.py
```

Latest results:

- `verify_page.js`: **PASS**. Default state 2011 Chinese GP, race 843, lap 31,
  VET focus, MAS ahead; 23 tower and probability rows; `#champions` renders all
  77 seasons with 16 `OPEN REPLAY` links, a working search filter, 77 loaded
  monochrome portraits and no overflow at 1560 px or 400 px; `OPEN REPLAY`
  switches back to `#replay` and loads the selected season finale. The circuit
  SVG has bbox 122.44 x 114 in the `0 0 220 130` viewBox, a start/finish tick
  and a 1.75px `currentColor` stroke; 23 replay portraits load monochrome. Zero
  console/page errors and zero external runtime requests; no horizontal overflow
  at 1560 px or 400 px; single-column layout at 400 px; light theme
  `rgb(255, 255, 255)` on `rgb(0, 0, 0)`; neutral bands present in all five
  time-based charts; no-result driver FIT is labelled and still has a
  probability; undercut sliders change the estimate and `Reset to live` restores
  the live values.
- `verify.py`: **PASS**. Page gaps and probabilities match an independent
  recomputation from the raw CSVs to better than `1e-6` (worst gap error 0.0 s,
  worst probability error `5.551e-17`). It also validates 77 champions across
  1950-2026 (69 with constructor champions, 16 with replay data), all 36 replay
  circuit layouts (92 era variants, 80-110 points per track), and all 108 local
  portraits as valid 160x160 RGBA PNGs.
- Manual pit-loss check: VET, 2011 Chinese GP, stop on lap 14:
  `111.161 + 122.384 - 2 x median(105.048, 105.136, 105.367) = 23.273 s`.
  The pipeline stores 23.273 s; the race median is 21.905 s and the displayed
  Shanghai circuit median is 21.791 s from 412 stops.
- Manual tyre-slope check: 2011 Chinese GP slope `0.119665 s/lap`, within-race
  `R² 0.57847`, `n=1145`. The page and JSON agree.
- Full-archive replay sweep: all 329 races at lap 1, mid-race and the final lap
  loaded with zero failures and zero console errors.

The screenshots in the repository are from the passing browser run, not mockups.

## Files

```text
build_data.py        data pipeline, champions, circuit layouts and model fit
fetch_drivers.py     one-time/offline portrait fetcher and PNG normaliser
index.html           the complete replay console and champions view
data/                index.json, model.json, s2011.json ... s2026.json
assets/drivers/      108 local 160x160 RGBA portraits keyed by driverRef
raw_circuits/        build-time cache: GeoJSON, SVG index and layout SVGs
verify.py            independent pandas/csv recomputation and asset validation
verify_page.js       headless Playwright checks and screenshots
README.md            this file
refresh.py           optional Kaggle download helper; not used by the page
```

`topics.md`, `package.json`, `package-lock.json` and the source CSVs are
inputs/context or verification-only; none are required at runtime.

## Known limits

- Neutralised laps are inferred from lap-time medians, not flags. A slow green
  lap can be shaded; a fast neutralised lap can be missed.
- Pit loss includes pit-lane traffic and driver variation. The source
  `milliseconds` field is lane time, not stationary time.
- The tyre regression is correlational. It does not observe compounds, fuel load,
  track temperature or rubber evolution; driver fixed effects are intercepts.
- The undercut model assumes independent pit-loss draws and a simple lap-noise
  term. It is a scenario estimate, not a race simulation.
- Circuit outlines are approximate: 80-110 simplified points per track and era
  variants keyed to known layout changes, not survey-grade geometry. A layout
  that fails to parse is cached as `fallback: true` and the card shows
  `LAYOUT UNAVAILABLE` rather than a broken path.
- Portraits are build-time snapshots from OpenF1/Wikipedia/Commons. A driver
  without a usable source keeps `has_img: false` and gets a monogram badge; no
  request is made at runtime.
- The 2026 championship standings in the archive stop at round 20 of 23, so the
  2026 champion row reflects the latest available standings, not a finished
  season. Pre-1958 rows have no constructor champion by definition.
- The archive is historical and ends with the races present in the CSV files.
  Rebuild after adding data to include new races.
