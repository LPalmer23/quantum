import React, { useState, useEffect } from "react";
import "./App.css";

type PricingResponse = {
  inputs: {
    S0: number;
    K: number;
    r: number;
    sigma: number;
    T: number;
    N_steps: number;
    shots: number;
    calibrate: boolean;
  };
  model_params: {
    dt: number;
    u: number;
    d: number;
    p_up: number;
    p_down: number;
    theta: number;
    n_p: number;
    num_bins: number;
  };
  outputs: {
    bs_call: number;
    q_call_raw: number;
    q_call_calibrated: number;
    E_ST_raw?: number;
    E_ST_calibrated?: number;
    discount_factor: number;
  };
  errors: {
    raw: { abs: number; rel: number };
    calibrated: { abs: number; rel: number };
  };
  histogram: { k: number; S: number; prob: number }[];
};

function App() {
  // Form state
  const [S0, setS0] = useState("100");
  const [K, setK] = useState("100");
  const [r, setR] = useState("0.02");
  const [sigma, setSigma] = useState("0.2");
  const [T, setT] = useState("5");
  const [NSteps, setNSteps] = useState("50");
  const [shots, setShots] = useState("4096");
  const [calibrate, setCalibrate] = useState(true);

  // Result + UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PricingResponse | null>(null);
  const [showIntro, setShowIntro] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setShowIntro(false), 1800); // 1.8s
    return () => clearTimeout(timer);
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload = {
        S0: Number(S0),
        K: Number(K),
        r: Number(r),
        sigma: Number(sigma),
        T: Number(T),
        N_steps: Number(NSteps),
        calibrate,
        shots: Number(shots),
      };

      const res = await fetch("http://127.0.0.1:8000/price", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server error ${res.status}: ${text}`);
      }

      const data: PricingResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message ?? "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  // One single return
  return (
    <>
      {/* Lens intro overlay */}
      {showIntro && (
        <div className="lens-overlay">
          <div className="lens-circle" />
          <p className="lens-text">Initializing quantum engine…</p>
        </div>
      )}

      {/* Main UI – only when intro finished */}
      {!showIntro && (
        <div className="app">
          <header className="header">
            <h1>ZyQ Labs – Quantum Option Pricer</h1>
            <p>
              Compare a classical Black–Scholes price with our quantum random
              walk engine. Change the inputs and see how the prices and errors
              move.
            </p>
          </header>

          <main className="layout">
            {/* Left: input form */}
            <section className="card">
              <h2>Inputs</h2>
              <form onSubmit={handleSubmit} className="form-grid">
                <label>
                  S₀ (initial price)
                  <input
                    type="number"
                    step="0.01"
                    value={S0}
                    onChange={(e) => setS0(e.target.value)}
                  />
                </label>

                <label>
                  K (strike)
                  <input
                    type="number"
                    step="0.01"
                    value={K}
                    onChange={(e) => setK(e.target.value)}
                  />
                </label>

                <label>
                  r (risk-free rate)
                  <input
                    type="number"
                    step="0.001"
                    value={r}
                    onChange={(e) => setR(e.target.value)}
                  />
                </label>

                <label>
                  σ (volatility)
                  <input
                    type="number"
                    step="0.01"
                    value={sigma}
                    onChange={(e) => setSigma(e.target.value)}
                  />
                </label>

                <label>
                  T (years to maturity)
                  <input
                    type="number"
                    step="0.25"
                    value={T}
                    onChange={(e) => setT(e.target.value)}
                  />
                </label>

                <label>
                  N_steps (time steps)
                  <input
                    type="number"
                    step="1"
                    value={NSteps}
                    onChange={(e) => setNSteps(e.target.value)}
                  />
                </label>

                <label>
                  Shots (simulator runs)
                  <input
                    type="number"
                    step="1"
                    value={shots}
                    onChange={(e) => setShots(e.target.value)}
                  />
                </label>

                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={calibrate}
                    onChange={(e) => setCalibrate(e.target.checked)}
                  />
                  Calibrate quantum walk to match E[S<sub>T</sub>]
                </label>

                <button type="submit" disabled={loading}>
                  {loading ? "Running quantum walk…" : "Run Quantum Pricing"}
                </button>
              </form>

              {error && <p className="error">⚠️ {error}</p>}
            </section>

            {/* Right: results */}
            <section className="card">
              <h2>Results</h2>
              {!result && !loading && (
                <p className="muted">
                  Run the pricer to see Black–Scholes vs quantum results.
                </p>
              )}

              {result && (
                <>
                  <div className="results-grid">
                    <div className="stat">
                      <h3>Black–Scholes Call</h3>
                      <p className="big">
                        ${result.outputs.bs_call.toFixed(2)}
                      </p>
                      <p className="muted">
                        Classical closed-form solution
                      </p>
                    </div>

                    <div className="stat">
                      <h3>Quantum Call (calibrated)</h3>
                      <p className="big">
                        ${result.outputs.q_call_calibrated.toFixed(2)}
                      </p>
                      <p className="muted">
                        After matching E[S<sub>T</sub>] to Black–Scholes
                      </p>
                    </div>

                    <div className="stat">
                      <h3>Calibration Error</h3>
                      <p className="big">
                        {result.errors.calibrated.abs.toFixed(3)}
                      </p>
                      <p className="muted">
                        |C<sub>quantum</sub> − C<sub>BS</sub>| (absolute)
                      </p>
                    </div>

                    <div className="stat">
                      <h3>Relative Error</h3>
                      <p className="big">
                        {(result.errors.calibrated.rel * 100).toFixed(2)}%
                      </p>
                      <p className="muted">
                        Relative to Black–Scholes price
                      </p>
                    </div>
                  </div>

                  <h3>Sample of Terminal Price Distribution</h3>
                  <p className="muted">
                    These are a few of the price bins from the quantum walk:
                  </p>
                  <table className="histogram">
                    <thead>
                      <tr>
                        <th>k (bin)</th>
                        <th>
                          S<sub>k</sub>
                        </th>
                        <th>Probability</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.histogram.slice(0, 12).map((bin) => (
                        <tr key={bin.k}>
                          <td>{bin.k}</td>
                          <td>{bin.S.toFixed(2)}</td>
                          <td>{bin.prob.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </section>
          </main>
        </div>
      )}
    </>
  );
}

export default App;
