#! /bin/bash
#
# Use:
#   ./run.sh [--eval-test] <clasificador> [carpeta_extract_forzada]
#
# clasificador: baseline | multinomial_nb | complement_nb |
#               logistic_regression | linear_svc | all
#
# Opciones:
#   --eval-test   Extrae (si falta cache) también data/test → feature_sets/*/test.cod
#                 y, tras cada modelo, predice + escribe stats/test.stats frente a
#                 lab_resources/DDI/data/test
#
# Sin carpeta_extract: mismo extractor features_v0 por defecto.
# Con segunda posición ( tras filtrar --eval-test ): models/<carpeta>/extract-features.py
#
# Ejemplos:
#   ./run.sh linear_svc
#   ./run.sh all features_v9
#   ./run.sh --eval-test logistic_regression features_v7

BASEDIR=./lab_resources/DDI
OUTDIR=outputs

EVAL_TEST=0
POSITIONAL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --eval-test) EVAL_TEST=1 ;;
        *) POSITIONAL_ARGS+=("$arg") ;;
    esac
done

MODEL="${POSITIONAL_ARGS[0]:-baseline}"
FEAT_OVERRIDE="${POSITIONAL_ARGS[1]:-}"

#
# ---------- Mapa: clasificador -> carpeta bajo models/ (con extract-features.py) ----------
#
feature_folder_for_classifier() {
    case "$1" in
        baseline)
            echo "features_v0"
            ;;
        multinomial_nb)
            echo "features_v0"
            ;;
        complement_nb)
            echo "features_v0"
            ;;
        logistic_regression)
            echo "features_v0"
            ;;
        linear_svc)
            echo "features_v0"
            ;;
        *)
            echo ""
            ;;
    esac
}

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
elif command -v py >/dev/null 2>&1; then
    PYTHON=py
else
    echo "Python not found"
    exit 1
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

resolve_extract_sources() {
    local FEAT_ARG="$1"

    if [ -f "models/${FEAT_ARG}/extract-features.py" ]; then
        EXTRACT_PY="models/${FEAT_ARG}/extract-features.py"
        FEATURE_KEY="${FEAT_ARG}"
    elif [ -f "${FEAT_ARG}/extract-features.py" ]; then
        EXTRACT_PY="${FEAT_ARG}/extract-features.py"
        FEATURE_KEY="$(basename "$(dirname "$EXTRACT_PY")")"
    else
        echo "ERROR: extract-features.py no encontrado para: ${FEAT_ARG}" >&2
        echo "Probado: models/${FEAT_ARG}/extract-features.py y ${FEAT_ARG}/extract-features.py" >&2
        exit 1
    fi

    FEATURESDIR="$OUTDIR/feature_sets/${FEATURE_KEY}"
    mkdir -p "$FEATURESDIR"
}

extract_features() {
    local cached_base=false
    local want_test=false
    local cached_test=false

    if [ -s "$FEATURESDIR/train.cod" ] && \
       [ -s "$FEATURESDIR/train.cod.cl" ] && \
       [ -s "$FEATURESDIR/devel.cod" ]; then
        cached_base=true
    fi
    [ "$EVAL_TEST" = "1" ] && want_test=true
    [ -s "$FEATURESDIR/test.cod" ] && cached_test=true

    if $cached_base && { [ "$want_test" = false ] || $cached_test; }; then
        echo "Features ya cacheadas (${FEATURE_KEY})"
        wc -l "$FEATURESDIR/train.cod" "$FEATURESDIR/train.cod.cl" "$FEATURESDIR/devel.cod"
        md5sum "$FEATURESDIR/train.cod" "$FEATURESDIR/train.cod.cl" "$FEATURESDIR/devel.cod" 2>/dev/null || \
            echo "(md5sum no disponible; omitiendo checksum)"
        if $want_test && $cached_test; then
            wc -l "$FEATURESDIR/test.cod"
            md5sum "$FEATURESDIR/test.cod" 2>/dev/null || true
        fi
        return
    fi

    if $cached_base && $want_test && ! $cached_test; then
        echo "Extracción sólo TEST (${EXTRACT_PY} -> ${FEATURESDIR}/test.cod)"
        ./corenlp-server.sh -quiet true -port 9000 -timeout 15000 &
        sleep 3
        $PYTHON "$EXTRACT_PY" "$BASEDIR/data/test/" >"$FEATURESDIR/test.cod"
        kill "$(cat /tmp/corenlp-server.running)" 2>/dev/null || true
        wc -l "$FEATURESDIR/test.cod"
        md5sum "$FEATURESDIR/test.cod" 2>/dev/null || true
        return
    fi

    rm -f "$FEATURESDIR/train.cod" "$FEATURESDIR/train.cod.cl" "$FEATURESDIR/devel.cod"
    [ "$EVAL_TEST" = "1" ] && rm -f "$FEATURESDIR/test.cod"

    ./corenlp-server.sh -quiet true -port 9000 -timeout 15000 &
    sleep 3

    echo "Extracción devel (${EXTRACT_PY} -> ${FEATURESDIR})"
    $PYTHON "$EXTRACT_PY" "$BASEDIR/data/devel/" >"$FEATURESDIR/devel.cod"

    echo "Extracción train (${EXTRACT_PY} -> ${FEATURESDIR})"
    $PYTHON "$EXTRACT_PY" "$BASEDIR/data/train/" \
        | tee "$FEATURESDIR/train.cod" \
        | cut -f4- >"$FEATURESDIR/train.cod.cl"

    if [ "$EVAL_TEST" = "1" ]; then
        echo "Extracción test (${EXTRACT_PY} -> ${FEATURESDIR})"
        $PYTHON "$EXTRACT_PY" "$BASEDIR/data/test/" >"$FEATURESDIR/test.cod"
    fi

    kill "$(cat /tmp/corenlp-server.running)" 2>/dev/null || true

    echo "Features generated:"
    wc -l "$FEATURESDIR/train.cod" "$FEATURESDIR/train.cod.cl" "$FEATURESDIR/devel.cod"
    md5sum "$FEATURESDIR/train.cod" "$FEATURESDIR/train.cod.cl" "$FEATURESDIR/devel.cod" 2>/dev/null || true
    if [ "$EVAL_TEST" = "1" ]; then
        wc -l "$FEATURESDIR/test.cod"
        md5sum "$FEATURESDIR/test.cod" 2>/dev/null || true
    fi
}

run_model() {
    NAME="$1"

    MODEL_OUT="$OUTDIR/runs/$FEATURE_KEY/$NAME"

    SAVEDIR=$MODEL_OUT/model
    PREDIR=$MODEL_OUT/predictions
    STATSDIR=$MODEL_OUT/stats
    PLOTDIR=$STATSDIR/plots

    mkdir -p $SAVEDIR $PREDIR $STATSDIR $PLOTDIR

    echo "Training: $NAME (features=${FEATURE_KEY})"

    $PYTHON utils/sklearn_train.py \
        $SAVEDIR/model.joblib \
        $SAVEDIR/vectorizer.joblib \
        $STATSDIR/cv_results.json \
        $PLOTDIR \
        "$NAME" \
        <"$FEATURESDIR/train.cod.cl"

    echo "Prediction train"

    $PYTHON utils/sklearn_predict.py \
        $SAVEDIR/model.joblib \
        $SAVEDIR/vectorizer.joblib \
        <"$FEATURESDIR/train.cod" \
        >"$PREDIR/train.out"

    echo "Prediction devel"

    $PYTHON utils/sklearn_predict.py \
        $SAVEDIR/model.joblib \
        $SAVEDIR/vectorizer.joblib \
        <"$FEATURESDIR/devel.cod" \
        >"$PREDIR/devel.out"

    echo "Evaluation train"

    $PYTHON ../evaluator.py DDI \
        $BASEDIR/data/train/ \
        $PREDIR/train.out \
        >"$STATSDIR/train.stats"

    echo "Evaluation devel"

    $PYTHON ../evaluator.py DDI \
        $BASEDIR/data/devel/ \
        $PREDIR/devel.out \
        >"$STATSDIR/devel.stats"

    if [ "$EVAL_TEST" = "1" ]; then
        if [ ! -s "$FEATURESDIR/test.cod" ]; then
            echo "ERROR: --eval-test pero no existe ${FEATURESDIR}/test.cod" >&2
            exit 1
        fi

        echo "Prediction test"

        $PYTHON utils/sklearn_predict.py \
            $SAVEDIR/model.joblib \
            $SAVEDIR/vectorizer.joblib \
            <"$FEATURESDIR/test.cod" \
            >"$PREDIR/test.out"

        echo "Evaluation test"

        $PYTHON ../evaluator.py DDI \
            $BASEDIR/data/test/ \
            $PREDIR/test.out \
            >"$STATSDIR/test.stats"
    fi
}

run_stack_for_classifier() {
    local NAME="$1"
    local KEY

    if [ -n "$FEAT_OVERRIDE" ]; then
        KEY="$FEAT_OVERRIDE"
    else
        KEY="$(feature_folder_for_classifier "$NAME")"
        if [ -z "$KEY" ]; then
            echo "ERROR: no hay carpeta de features definida para «$NAME». Añádela en feature_folder_for_classifier() en run.sh" >&2
            exit 1
        fi
    fi

    resolve_extract_sources "$KEY"
    extract_features
    run_model "$NAME"
}

if [ "$MODEL" = "all" ]; then
    if [ -n "$FEAT_OVERRIDE" ]; then
        resolve_extract_sources "$FEAT_OVERRIDE"
        extract_features
        run_model multinomial_nb
        run_model complement_nb
        run_model logistic_regression
        run_model linear_svc
    else
        run_stack_for_classifier multinomial_nb
        run_stack_for_classifier complement_nb
        run_stack_for_classifier logistic_regression
        run_stack_for_classifier linear_svc
    fi
else
    run_stack_for_classifier "$MODEL"
fi
