import React, { useMemo, useRef, useState } from "react";
import { Activity, Brain, FileText, Play, ShieldCheck } from "lucide-react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const examples = [
  "Analyze https://github.com/langchain-ai/langgraph and generate an interview-ready architecture report.",
  "Analyze the skill gaps for a ByteDance Agent infrastructure internship JD.",
  "Design fact-checking rules and replan triggers for a CriticAgent.",
];

function stageIcon(stage) {
  if (stage.includes("critic")) return <ShieldCheck size={16} />;
  if (stage.includes("draft") || stage.includes("done")) return <FileText size={16} />;
  if (stage.includes("eval")) return <Activity size={16} />;
  return <Brain size={16} />;
}

function App() {
  const [query, setQuery] = useState(examples[0]);
  const [events, setEvents] = useState([]);
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const socketRef = useRef(null);

  const score = useMemo(() => result?.eval?.total_score ?? null, [result]);

  function runTask() {
    setEvents([]);
    setResult(null);
    setRunning(true);

    const socket = new WebSocket("ws://127.0.0.1:8000/ws/run");
    socketRef.current = socket;

    socket.onopen = () => {
      socket.send(JSON.stringify({ query, task_type: "research_report", sources: [] }));
    };

    socket.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (event.stage === "done") {
        setResult(event.result);
        setRunning(false);
        socket.close();
        return;
      }
      setEvents((current) => [...current, event]);
    };

    socket.onerror = () => setRunning(false);
    socket.onclose = () => setRunning(false);
  }

  return (
    <main className="app">
      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>DevResearch Agent</h1>
            <p>Multi-agent research, report generation, and evaluation.</p>
          </div>
          <button className="runButton" onClick={runTask} disabled={running}>
            <Play size={16} />
            {running ? "Running" : "Run"}
          </button>
        </header>

        <div className="queryPanel">
          <textarea value={query} onChange={(event) => setQuery(event.target.value)} />
          <div className="chips">
            {examples.map((item) => (
              <button key={item} onClick={() => setQuery(item)}>
                {item}
              </button>
            ))}
          </div>
        </div>

        <section className="grid">
          <div className="panel">
            <h2>Trace</h2>
            <div className="timeline">
              {events.map((event, index) => (
                <article className="event" key={`${event.stage}-${index}`}>
                  <div className="eventIcon">{stageIcon(event.stage)}</div>
                  <div>
                    <strong>{event.stage}</strong>
                    <p>{event.message}</p>
                  </div>
                </article>
              ))}
              {events.length === 0 && <p className="empty">Waiting for a task.</p>}
            </div>
          </div>

          <div className="panel">
            <h2>Evaluation</h2>
            {score ? (
              <div className="score">
                <span>{score}</span>
                <small>/ 5</small>
                <ul>
                  <li>Tool calls: {result.eval.tool_call_accuracy}</li>
                  <li>Citation support: {result.eval.citation_support}</li>
                  <li>Report completeness: {result.eval.report_completeness}</li>
                  <li>Factual consistency: {result.eval.factual_consistency}</li>
                </ul>
              </div>
            ) : (
              <p className="empty">The score appears after the task finishes.</p>
            )}
          </div>
        </section>

        <section className="panel report">
          <h2>Report</h2>
          <pre>{result?.report || "The final report will appear here."}</pre>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
