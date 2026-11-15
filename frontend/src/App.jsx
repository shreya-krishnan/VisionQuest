import { useEffect, useRef, useState } from "react";
import Navigation from "./components/Navigation.jsx";
import Overview from "./components/Overview.jsx";
import Workspace from "./components/Workspace.jsx";
import Results from "./components/Results.jsx";
import Evaluation from "./components/Evaluation.jsx";
import Methodology from "./components/Methodology.jsx";
import Footer from "./components/Footer.jsx";
import { initializePrototypeInteractions } from "./interactions.js";
import { renderVisionQuest } from "./renderVisionQuest.js";
import useVisionQuest from "./hooks/useVisionQuest.js";
import "./prototype.css";

export default function App() {
  const pageRef = useRef(null);
  const actionsRef = useRef({});
  const [view, setView] = useState(() => {
    const initial = location.hash.slice(1);
    return ["home", "workspace", "results", "evaluation", "method"].includes(initial)
      ? initial
      : "home";
  });
  const visionQuest = useVisionQuest();
  actionsRef.current = visionQuest;

  useEffect(() => initializePrototypeInteractions(pageRef.current, {
    onFileSelected: (file) => actionsRef.current.chooseFile(file),
    onMethodChange: (method) => actionsRef.current.changeMethod(method),
    onRun: async (method) => {
      const result = await actionsRef.current.run(method);
      return result;
    },
    onReset: () => actionsRef.current.resetSession(),
    onNavigate: navigate,
  }), []);

  function navigate(nextView) {
    const normalized = nextView === "overview" ? "home" : nextView;
    if (!["home", "workspace", "results", "evaluation", "method"].includes(normalized)) return;
    setView(normalized);
    if (location.hash !== `#${normalized}`) history.pushState({ view: normalized }, "", `#${normalized}`);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  useEffect(() => {
    const sync = () => {
      const next = location.hash.slice(1);
      setView(["home", "workspace", "results", "evaluation", "method"].includes(next) ? next : "home");
      window.scrollTo({ top: 0, behavior: "instant" });
    };
    window.addEventListener("popstate", sync);
    window.addEventListener("hashchange", sync);
    return () => {
      window.removeEventListener("popstate", sync);
      window.removeEventListener("hashchange", sync);
    };
  }, []);

  useEffect(() => {
    const nav = pageRef.current?.querySelector(".nav");
    const sectionId = view === "home" ? "overview" : view;
    const section = pageRef.current?.querySelector(`#${sectionId}`);
    if (!nav || !section) return;
    nav.style.setProperty("--nav-bg", section.dataset.navBg);
    nav.style.setProperty("--nav-bg-solid", section.dataset.navSolid);
    nav.style.setProperty("--nav-ink", section.dataset.navInk);
    nav.querySelectorAll("[data-target]").forEach((link) => {
      link.setAttribute("aria-current", link.dataset.target === view ? "true" : "false");
    });
  }, [view]);

  useEffect(() => {
    if (pageRef.current) renderVisionQuest(pageRef.current, visionQuest);
  }, [visionQuest]);

  return (
    <div ref={pageRef} data-view={view}>
      <div className="wipe" aria-hidden="true" />
      <Navigation />
      <main>
        <Overview />
        <Workspace />
        <Results />
        <Evaluation />
        <Methodology />
      </main>
      {(["home", "method"].includes(view)) && <Footer />}
    </div>
  );
}
