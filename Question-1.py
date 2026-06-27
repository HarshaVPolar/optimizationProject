import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Settings
randomState = 4
testSize = 0.30
dataPath = "train.csv"

df = pd.read_csv(dataPath, parse_dates=["datetime"])

if "casual" in df.columns and "registered" in df.columns:
    df = df.drop(columns=["casual", "registered"])

if "datetime" in df.columns:
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

featureCols = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "dayofweek", "month", "year"
]

xAll = df[featureCols].copy()
yAll = df["count"].values

n = len(df)
np.random.seed(randomState)
perm = np.random.permutation(n)
splitIdx = int((1 - testSize) * n)
trainIdx = perm[:splitIdx]
testIdx = perm[splitIdx:]

xTrainDf = xAll.iloc[trainIdx].reset_index(drop=True)
xTestDf = xAll.iloc[testIdx].reset_index(drop=True)
yTrain = yAll[trainIdx].astype(np.float64)
yTest = yAll[testIdx].astype(np.float64)

numFeats = ["temp", "atemp", "humidity", "windspeed"]
catFeats = ["season", "holiday", "workingday", "weather", "hour", "dayofweek", "month", "year"]

numMeans = xTrainDf[numFeats].mean(axis=0).values
numStds = xTrainDf[numFeats].std(axis=0).values
numStds[numStds == 0] = 1.0

xTrainNum = ((xTrainDf[numFeats].values - numMeans) / numStds).astype(np.float64)
xTestNum = ((xTestDf[numFeats].values - numMeans) / numStds).astype(np.float64)

trainCat = pd.get_dummies(xTrainDf[catFeats], columns=catFeats)
testCat = pd.get_dummies(xTestDf[catFeats], columns=catFeats)
testCat = testCat.reindex(columns=trainCat.columns, fill_value=0)

trainCatVals = trainCat.values.astype(np.float64)
testCatVals = testCat.values.astype(np.float64)

xTrain = np.concatenate([xTrainNum, trainCatVals], axis=1).astype(np.float64)
xTest = np.concatenate([xTestNum, testCatVals], axis=1).astype(np.float64)

def addIntercept(X):
    return np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)

def solveLeastSquares(X_design, y):
    Xd = np.asarray(X_design, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    try:
        return np.linalg.lstsq(Xd, y_arr, rcond=None)[0]
    except Exception:
        return np.linalg.pinv(Xd) @ y_arr

def predict(X_design, w):
    return np.asarray(X_design, dtype=np.float64) @ np.asarray(w, dtype=np.float64)

def mse(y_true, y_pred):
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return np.mean((y_t - y_p) ** 2)

def r2Score(y_true, y_pred):
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def evaluate(name, y_true, y_pred):
    try:
        y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    except Exception:
        print(f"{name:<32} | ERROR: predictions cannot be converted to float64")
        return np.inf, -np.inf
    if np.any(np.isnan(y_pred_arr)) or np.any(np.isinf(y_pred_arr)):
        print(f"{name:<32} | ERROR: Invalid predictions (NaN/Inf).")
        return np.inf, -np.inf
    m = mse(y_true, y_pred_arr)
    r = r2Score(y_true, y_pred_arr)
    print(f"{name:<32} | MSE: {m:,.2f} | R²: {r:.4f}")
    return m, r

results = {}
allPreds = {}

print("\n--- Training Models ---\n")

# Linear
xTrainDesign = addIntercept(xTrain)
xTestDesign = addIntercept(xTest)
wLinear = solveLeastSquares(xTrainDesign, yTrain)
predLinear = predict(xTestDesign, wLinear).astype(np.float64)
results["Linear"] = evaluate("Linear Regression", yTest, predLinear)
allPreds["Linear"] = predLinear

# Polynomials (numeric only)
def addPolynomialFeaturesNumeric(xNum, degree):
    parts = [xNum]
    for d in range(2, degree + 1):
        parts.append(xNum ** d)
    return np.concatenate(parts, axis=1)

for deg in [2, 3, 4]:
    polyTrainNum = addPolynomialFeaturesNumeric(xTrainNum, deg)
    polyTestNum = addPolynomialFeaturesNumeric(xTestNum, deg)
    polyMean = polyTrainNum.mean(axis=0)
    polyStd = polyTrainNum.std(axis=0)
    polyStd[polyStd == 0] = 1.0
    polyTrainNumScaled = ((polyTrainNum - polyMean) / polyStd).astype(np.float64)
    polyTestNumScaled = ((polyTestNum - polyMean) / polyStd).astype(np.float64)
    xTrainPoly = np.concatenate([polyTrainNumScaled, trainCatVals], axis=1).astype(np.float64)
    xTestPoly = np.concatenate([polyTestNumScaled, testCatVals], axis=1).astype(np.float64)
    xTrainPolyDesign = addIntercept(xTrainPoly)
    xTestPolyDesign = addIntercept(xTestPoly)
    wPoly = solveLeastSquares(xTrainPolyDesign, yTrain)
    predPoly = predict(xTestPolyDesign, wPoly).astype(np.float64)
    results[f"Poly_deg{deg}"] = evaluate(f"Polynomial d={deg} (numeric-only, no interactions)", yTest, predPoly)
    allPreds[f"Poly_deg{deg}"] = predPoly

# Quadratic with numeric interactions
def addQuadraticInteractionsNumeric(xNum):
    n_samples, n_features = xNum.shape
    parts = []
    for i in range(n_features):
        for j in range(i, n_features):
            parts.append((xNum[:, i] * xNum[:, j]).reshape(-1, 1))
    if len(parts) == 0:
        return np.zeros((n_samples, 0), dtype=np.float64)
    return np.concatenate(parts, axis=1).astype(np.float64)

xTrainInter = addQuadraticInteractionsNumeric(xTrainNum)
xTestInter = addQuadraticInteractionsNumeric(xTestNum)

print(f"Interaction features: Train {xTrainInter.shape}, Test {xTestInter.shape}")

if xTrainInter.shape[1] > 0:
    interMean = xTrainInter.mean(axis=0)
    interStd = xTrainInter.std(axis=0)
    interStd[interStd == 0] = 1.0
    xTrainInterScaled = ((xTrainInter - interMean) / interStd).astype(np.float64)
    xTestInterScaled = ((xTestInter - interMean) / interStd).astype(np.float64)
else:
    xTrainInterScaled = xTrainInter
    xTestInterScaled = xTestInter

xTrainQuad = np.concatenate([xTrainNum, xTrainInterScaled, trainCatVals], axis=1).astype(np.float64)
xTestQuad = np.concatenate([xTestNum, xTestInterScaled, testCatVals], axis=1).astype(np.float64)

print(f"\nFinal quadratic feature matrix: Train {xTrainQuad.shape}, Test {xTestQuad.shape}")

xTrainQuadDesign = addIntercept(xTrainQuad)
xTestQuadDesign = addIntercept(xTestQuad)
wQuad = solveLeastSquares(xTrainQuadDesign, yTrain)
predQuad = predict(xTestQuadDesign, wQuad).astype(np.float64)
results["Quadratic_interactions"] = evaluate("Quadratic (numeric-only interactions)", yTest, predQuad)
allPreds["Quadratic_interactions"] = predQuad

# Summary
print("\n--- Model Summary (sorted by MSE) ---\n")
summaryDf = pd.DataFrame.from_dict(results, orient="index", columns=["MSE", "R2"]).sort_values("MSE")
print(summaryDf)
bestModel = summaryDf.index[0]
print(f"\nBest Model → {bestModel}")

# Plot (clean, minimal)
bestPred = allPreds[bestModel]
plt.figure(figsize=(8, 8))
plt.scatter(yTest, bestPred, alpha=0.45, s=22, edgecolors='black', linewidth=0.4)
maxVal = max(yTest.max(), float(bestPred.max()))
plt.plot([0, maxVal], [0, maxVal], linestyle='--', linewidth=2, color='#cc0000', label='Ideal Prediction')
plt.xlabel("Actual Count", fontsize=12, fontweight="bold")
plt.ylabel("Predicted Count", fontsize=12, fontweight="bold")
plt.title(
    f"{bestModel.replace('_', ' ').title()}\nMSE = {results[bestModel][0]:,.2f}   |   R² = {results[bestModel][1]:.4f}",
    fontsize=13, fontweight="bold", pad=10
)
plt.legend(frameon=True, fontsize=10)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
