/* =========================================================
   MANGANESE EXPLORATION ANALYTICS
========================================================= */


/* =========================================================
   PROSPECTIVITY COLORS
========================================================= */

const COLORS = {

    LOW: "#3aa069",

    MODERATE: "#dfad2b",

    HIGH: "#d45248"

};


/* =========================================================
   GLOBAL VARIABLES
========================================================= */

let occurrenceData = [];

let occurrenceLayer = null;

let prospectivityLayer = null;

let metrics = {};

let currentBestModel = null;


/* =========================================================
   HELPER
========================================================= */

const $ = id => document.getElementById(id);


/* =========================================================
   MESSAGE
========================================================= */

function setMessage(text, type = "error") {

    const box = $("message");

    if (!text) {

        box.classList.add("hidden");

        box.textContent = "";

        return;
    }

    box.classList.remove("hidden");

    box.className = "message";

    if (type === "ok") {

        box.classList.add("ok");

    }

    box.textContent = text;
}


/* =========================================================
   STATUS
========================================================= */

function setStatus(
    dotId,
    textId,
    state,
    text
) {

    const dot = $(dotId);

    const label = $(textId);

    dot.className = "status-dot";

    if (state === "ready") {

        dot.classList.add("ready");

    }

    if (state === "error") {

        dot.classList.add("error");

    }

    label.textContent = text;
}


/* =========================================================
   API
========================================================= */

async function api(url, options = {}) {

    const response = await fetch(url, options);

    let data;

    try {

        data = await response.json();

    } catch {

        throw new Error(
            "The server returned an invalid response."
        );

    }

    if (!response.ok) {

        throw new Error(
            data.error || "Request failed."
        );

    }

    return data;
}


async function post(url, body = {}) {

    return api(url, {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify(body)

    });

}


/* =========================================================
   MAPS
========================================================= */

const occurrenceMap =
    L.map("occurrenceMap").setView(
        [21, 82],
        5
    );


L.tileLayer(
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        attribution:
            "&copy; OpenStreetMap contributors"
    }
).addTo(occurrenceMap);


const map =
    L.map("map").setView(
        [21, 82],
        5
    );


L.tileLayer(
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        attribution:
            "&copy; OpenStreetMap contributors"
    }
).addTo(map);


occurrenceLayer =
    L.layerGroup().addTo(occurrenceMap);


/* =========================================================
   SELECTED STATE
========================================================= */

function getSelectedState() {

    return $("state").value.trim();

}


/* =========================================================
   ESCAPE HTML
========================================================= */

function escapeHTML(value) {

    return String(value)

        .replaceAll("&", "&amp;")

        .replaceAll("<", "&lt;")

        .replaceAll(">", "&gt;")

        .replaceAll('"', "&quot;")

        .replaceAll("'", "&#039;");
}


/* =========================================================
   OCCURRENCE MAP
========================================================= */

function updateOccurrenceMap() {

    occurrenceLayer.clearLayers();

    const selected =
        getSelectedState();

    let points = occurrenceData;


    if (selected) {

        points =
            occurrenceData.filter(
                p =>
                    String(p.state)
                        .trim()
                        .toLowerCase() ===
                    selected.toLowerCase()
            );

    }


    $("occCount").textContent =
        occurrenceData.length;

    $("selectedCount").textContent =
        points.length;


    points.forEach(point => {

        const marker =
            L.circleMarker(
                [point.lat, point.lon],
                {

                    radius: 5,

                    color: "#7d1717",

                    weight: 1,

                    fillColor: "#a92222",

                    fillOpacity: 0.9

                }
            );


        marker.bindPopup(`

            <strong>
                ${escapeHTML(
                    point.locality ||
                    "Manganese occurrence"
                )}
            </strong>

            <br>

            State:
            ${escapeHTML(
                point.state ||
                "Unknown"
            )}

            <br>

            Host rock:
            ${escapeHTML(
                point.host_rock ||
                "Not available"
            )}

        `);


        marker.addTo(
            occurrenceLayer
        );

    });


    if (points.length > 0) {

        const bounds =
            L.latLngBounds(
                points.map(
                    p => [p.lat, p.lon]
                )
            );


        occurrenceMap.fitBounds(
            bounds,
            {
                padding: [20, 20]
            }
        );

    }

}


/* =========================================================
   STATE DROPDOWN
========================================================= */

function populateStates() {

    const select =
        $("state");

    const counts = {};


    occurrenceData.forEach(point => {

        const state =
            String(
                point.state || ""
            ).trim();


        if (!state) {

            return;

        }


        counts[state] =
            (counts[state] || 0) + 1;

    });


    Object.keys(counts)
        .sort()
        .forEach(state => {

            const option =
                document.createElement(
                    "option"
                );


            option.value =
                state;


            option.textContent =
                `${state} (${counts[state]} occurrences)`;


            select.appendChild(option);

        });

}


/* =========================================================
   STATE HELP
========================================================= */

function updateStateHelp() {

    const selected =
        getSelectedState();


    if (!selected) {

        $("stateHelp").textContent =
            "All available GSI occurrences will be used.";

        $("analyzeBtn").disabled = false;

        return;
    }


    const count =
        occurrenceData.filter(
            p =>
                String(p.state)
                    .trim()
                    .toLowerCase() ===
                selected.toLowerCase()
        ).length;


    if (count < 3) {

        $("stateHelp").textContent =
            `Only ${count} occurrence available. Select another study area.`;

        $("analyzeBtn").disabled = true;

    } else {

        $("stateHelp").textContent =
            `${count} known GSI occurrences available in this study area.`;

        $("analyzeBtn").disabled = false;

    }

}


/* =========================================================
   PROGRESS
========================================================= */

function showProgress() {

    $("progressPanel")
        .classList.remove("hidden");

}


function updateProgress(
    percent,
    title,
    text,
    activeStep
) {

    $("progressBar")
        .style.width =
        `${percent}%`;

    $("progressPercent")
        .textContent =
        `${percent}%`;

    $("progressTitle")
        .textContent =
        title;

    $("progressText")
        .textContent =
        text;


    const steps = [

        "stepFeatures",

        "stepModel",

        "stepPrediction",

        "stepMap"

    ];


    steps.forEach(
        (id, index) => {

            const element = $(id);

            element.classList.remove(
                "active",
                "done"
            );


            if (index < activeStep) {

                element.classList.add(
                    "done"
                );

            }


            if (index === activeStep) {

                element.classList.add(
                    "active"
                );

            }

        }
    );

}


function finishProgress() {

    $("progressBar")
        .style.width = "100%";

    $("progressPercent")
        .textContent = "100%";

    $("progressTitle")
        .textContent =
        "Analysis completed";

    $("progressText")
        .textContent =
        "Prospectivity map is ready.";


    [
        "stepFeatures",
        "stepModel",
        "stepPrediction",
        "stepMap"
    ].forEach(id => {

        $(id)
            .classList.remove(
                "active"
            );

        $(id)
            .classList.add(
                "done"
            );

    });

}


/* =========================================================
   LOAD STATUS
========================================================= */

async function loadStatus() {

    const status =
        await api("/api/status");


    /* OCCURRENCE */

    if (status.dataset_present) {

        setStatus(
            "occStatusDot",
            "occStatus",
            "ready",
            "Ready"
        );

    } else {

        setStatus(
            "occStatusDot",
            "occStatus",
            "error",
            "Missing"
        );

    }


    /* SATELLITE */

    $("period").textContent =
        `${status.s2_period[0]} → ${status.s2_period[1]}`;


    if (status.ee_project_set) {

        setStatus(
            "satStatusDot",
            "satStatus",
            "ready",
            "Configured"
        );

    } else {

        setStatus(
            "satStatusDot",
            "satStatus",
            "error",
            "Not configured"
        );

    }


    /* MODEL */

    if (status.metrics_ready) {

        setStatus(
            "modelStatusDot",
            "modelStatus",
            "ready",
            "Ready"
        );

    } else {

        setStatus(
            "modelStatusDot",
            "modelStatus",
            "error",
            "Not trained"
        );

    }


    /* MAP */

    if (status.prediction_ready) {

        setStatus(
            "resultStatusDot",
            "resultStatus",
            "ready",
            "Available"
        );

    } else {

        setStatus(
            "resultStatusDot",
            "resultStatus",
            "error",
            "Not generated"
        );

    }


    if (status.message) {

        setMessage(
            status.message
        );

    }


    return status;

}


/* =========================================================
   LOAD OCCURRENCES
========================================================= */

async function loadOccurrences() {

    const data =
        await api(
            "/api/occurrences"
        );


    occurrenceData =
        data.points || [];


    populateStates();

    updateOccurrenceMap();

    updateStateHelp();

}


/* =========================================================
   BUILD FEATURES
========================================================= */

async function buildFeatures(state) {

    updateProgress(

        10,

        "Preparing satellite analysis",

        "Collecting known occurrences and background locations...",

        0

    );


    const result =
        await post(
            "/api/build-features",
            {
                state: state
            }
        );


    updateProgress(

        35,

        "Satellite feature extraction completed",

        `${result.positives_with_satellite_data} known occurrence points and ${result.background_with_satellite_data} background points processed.`,

        1

    );


    return result;

}


/* =========================================================
   TRAIN MODELS
========================================================= */

async function trainModels() {

    updateProgress(

        42,

        "Training AI models",

        "Comparing machine learning models using spatial validation...",

        1

    );


    const result =
        await post(
            "/api/train",
            {
                models: [
                    "random_forest",
                    "svm",
                    "xgboost"
                ]
            }
        );


    metrics = result;


    /*
       We still use the model metrics internally
       to select the best model.

       They are NOT displayed in the frontend.
    */

    currentBestModel =
        chooseBestModel(result);


    /*
       Feature importance is still useful to
       understand what influenced the prediction.
    */

    if (
        currentBestModel &&
        result[currentBestModel]
    ) {

        displayFeatureImportance(
            result[currentBestModel]
        );

    }


    updateProgress(

        65,

        "AI model selected",

        `Best available model selected for prediction.`,

        2

    );


    return result;

}


/* =========================================================
   CHOOSE BEST MODEL
========================================================= */

function chooseBestModel(result) {

    let best = null;

    let bestScore = -Infinity;


    Object.entries(result)
        .forEach(
            ([model, values]) => {

                const score =
                    values.roc_auc ??
                    values.f1 ??
                    values.accuracy ??
                    -Infinity;


                if (score > bestScore) {

                    bestScore =
                        score;

                    best =
                        model;

                }

            }
        );


    return best ||
        "random_forest";

}


/* =========================================================
   MODEL NAME
========================================================= */

function formatModelName(name) {

    if (!name) {

        return "AI Model";

    }


    return name

        .replaceAll(
            "_",
            " "
        )

        .replace(
            /\b\w/g,
            c => c.toUpperCase()
        );

}


/* =========================================================
   PREDICTION
========================================================= */

async function runPrediction() {

    updateProgress(

        70,

        "Generating prospectivity map",

        "Applying the selected AI model across the study area...",

        2

    );


    const result =
        await post(
            "/api/predict",
            {
                model:
                    currentBestModel
            }
        );


    updateProgress(

        92,

        "Preparing exploration map",

        `${result.cells || 0} locations analyzed.`,

        3

    );


    displayPredictionSummary(
        result
    );


    await showProspectivityMap();


    finishProgress();


    setStatus(

        "modelStatusDot",

        "modelStatus",

        "ready",

        "Ready"

    );


    setStatus(

        "resultStatusDot",

        "resultStatus",

        "ready",

        "Available"

    );

}


/* =========================================================
   RESULT SUMMARY
========================================================= */

function displayPredictionSummary(
    result
) {

    const counts =
        result.counts || {};


    const low =
        counts.LOW ??
        counts.low ??
        0;


    const moderate =
        counts.MODERATE ??
        counts.moderate ??
        0;


    const high =
        counts.HIGH ??
        counts.high ??
        0;


    const total =
        result.cells ??
        (
            low +
            moderate +
            high
        );


    $("lowCount")
        .textContent =
        low.toLocaleString();


    $("moderateCount")
        .textContent =
        moderate.toLocaleString();


    $("highCount")
        .textContent =
        high.toLocaleString();


    $("totalCount")
        .textContent =
        total.toLocaleString();

}


/* =========================================================
   SHOW PROSPECTIVITY MAP
========================================================= */

async function showProspectivityMap() {

    const geojson =
        await api(
            "/api/prospectivity"
        );


    if (prospectivityLayer) {

        map.removeLayer(
            prospectivityLayer
        );

    }


    /*
       Color the GeoJSON polygons
       according to LOW / MODERATE / HIGH.
    */

    prospectivityLayer =
        L.geoJSON(

            geojson,

            {

                style: feature => {

                    const cls =
                        String(
                            feature.properties?.class ||
                            "LOW"
                        ).toUpperCase();


                    const color =
                        COLORS[cls] ||
                        COLORS.LOW;


                    return {

                        color: color,

                        weight: 0.5,

                        fillColor: color,

                        fillOpacity: 0.60

                    };

                },


                onEachFeature:
                    (feature, layer) => {

                        const cls =
                            String(
                                feature.properties?.class ||
                                "Unknown"
                            ).toUpperCase();


                        const probability =
                            feature.properties?.probability;


                        let popup = `

                            <strong>
                                ${escapeHTML(cls)}
                            </strong>
                            prospectivity

                        `;


                        if (
                            probability !==
                            undefined
                        ) {

                            popup += `

                                <br>
                                Probability:
                                ${probability}

                            `;

                        }


                        layer.bindPopup(
                            popup
                        );

                    }

            }

        ).addTo(map);


    const bounds =
        prospectivityLayer
            .getBounds();


    if (bounds.isValid()) {

        map.fitBounds(

            bounds,

            {
                padding: [20, 20]
            }

        );

    }


    /*
       Add known GSI occurrences
       on top of the prediction map.
    */

    const selected =
        getSelectedState();


    const points =
        selected

            ? occurrenceData.filter(
                p =>
                    String(p.state)
                        .trim()
                        .toLowerCase() ===
                    selected.toLowerCase()
            )

            : occurrenceData;


    points.forEach(point => {

        L.circleMarker(

            [point.lat, point.lon],

            {

                radius: 5,

                color: "#681313",

                weight: 1,

                fillColor: "#a92222",

                fillOpacity: 0.95

            }

        )

        .bindPopup(`

            <strong>
                Known GSI Occurrence
            </strong>

            <br>

            ${escapeHTML(
                point.locality || ""
            )}

            <br>

            ${escapeHTML(
                point.state || ""
            )}

        `)

        .addTo(map);

    });

}


/* =========================================================
   FEATURE IMPORTANCE
========================================================= */

function displayFeatureImportance(
    modelData
) {

    const container =
        $("featureImportance");


    if (
        !modelData ||
        !modelData.feature_importance
    ) {

        container.innerHTML = `

            <div class="empty-state">

                Feature importance is not
                available for this model.

            </div>

        `;

        return;

    }


    const entries =
        Object.entries(
            modelData.feature_importance
        )

        .sort(
            (a, b) =>
                b[1] - a[1]
        )

        .slice(
            0,
            8
        );


    if (!entries.length) {

        container.innerHTML = `

            <div class="empty-state">

                No feature importance data available.

            </div>

        `;

        return;

    }


    const max =
        Math.max(

            ...entries.map(
                item =>
                    Number(item[1]) || 0
            ),

            0.000001

        );


    container.innerHTML =

        entries

            .map(
                ([name, value]) => {

                    const numeric =
                        Number(value) || 0;


                    const width =
                        Math.max(
                            2,
                            (
                                numeric /
                                max
                            ) * 100
                        );


                    return `

                        <div class="feature-row">

                            <div class="feature-name">

                                <span>
                                    ${escapeHTML(name)}
                                </span>

                                <span>
                                    ${numeric.toFixed(3)}
                                </span>

                            </div>


                            <div class="feature-track">

                                <div
                                    class="feature-fill"
                                    style="
                                        width:${width}%
                                    "
                                ></div>

                            </div>

                        </div>

                    `;

                }

            )

            .join("");

}


/* =========================================================
   ANALYZE
========================================================= */

async function analyze() {

    const state =
        getSelectedState();


    if (

        state &&

        occurrenceData.filter(

            p =>

                String(p.state)
                    .trim()
                    .toLowerCase() ===
                state.toLowerCase()

        ).length < 3

    ) {

        setMessage(
            "This study area does not contain enough known occurrence points for analysis."
        );

        return;

    }


    $("analyzeBtn")
        .disabled = true;


    showProgress();

    setMessage("");


    try {


        /* STEP 1 */

        await buildFeatures(
            state
        );


        /* STEP 2 */

        await trainModels();


        /* STEP 3 */

        await runPrediction();


        /* REFRESH OCCURRENCES */

        updateOccurrenceMap();


        setMessage(

            "Analysis completed successfully. The prospectivity map is ready.",

            "ok"

        );


        $("resultsSection")
            .scrollIntoView({

                behavior: "smooth",

                block: "start"

            });


    } catch (error) {

        console.error(error);


        setMessage(

            error.message ||
            "Analysis failed. Please check the backend terminal."

        );


        $("progressTitle")
            .textContent =
            "Analysis could not be completed";


        $("progressText")
            .textContent =
            error.message ||
            "Unknown error.";


    } finally {

        $("analyzeBtn")
            .disabled = false;

    }

}


/* =========================================================
   INITIALIZATION
========================================================= */

async function init() {

    try {

        const status =
            await loadStatus();


        if (!status.dataset_present) {

            return;

        }


        await loadOccurrences();


        /*
           If an existing prediction is available,
           display it.
        */

        if (
            status.prediction_ready
        ) {

            try {

                await showProspectivityMap();

            } catch (error) {

                console.warn(
                    "Could not load saved prospectivity:",
                    error
                );

            }

        }

    } catch (error) {

        console.error(error);


        setMessage(

            error.message ||
            "Unable to connect to the Flask backend."

        );

    }

}


/* =========================================================
   EVENTS
========================================================= */

$("state")
    .addEventListener(
        "change",
        () => {

            updateOccurrenceMap();

            updateStateHelp();

        }
    );


$("analyzeBtn")
    .addEventListener(
        "click",
        analyze
    );


$("startBtn")
    .addEventListener(
        "click",
        () => {

            $("exploration")
                .scrollIntoView({

                    behavior: "smooth",

                    block: "start"

                });

        }
    );


/* =========================================================
   START
========================================================= */

init();