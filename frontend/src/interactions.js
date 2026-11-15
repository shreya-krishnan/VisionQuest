export function initializePrototypeInteractions(root, actions = {}) {
  const $ = (selector, scope = root) => scope.querySelector(selector);
  const $$ = (selector, scope = root) => [...scope.querySelectorAll(selector)];
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const cleanups = [];
  const timers = new Set();
  let scrollFrame = 0;

  const listen = (element, event, callback, options) => {
    element.addEventListener(event, callback, options);
    cleanups.push(() => element.removeEventListener(event, callback, options));
  };
  const later = (callback, delay) => {
    const timer = setTimeout(() => {
      timers.delete(timer);
      callback();
    }, delay);
    timers.add(timer);
  };

  const revealObserver = new IntersectionObserver(
    (entries) => entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in");
        revealObserver.unobserve(entry.target);
      }
    }),
    { threshold: 0.15, rootMargin: "0px 0px -6% 0px" },
  );
  $$("#overview .reveal, #overview .lines, #evaluation .reveal, #evaluation .lines, #method .reveal, #method .lines")
    .forEach((element) => revealObserver.observe(element));
  cleanups.push(() => revealObserver.disconnect());

  const overview = $("#overview");
  const firstFrame = requestAnimationFrame(() => overview?.classList.add("in-view"));
  cleanups.push(() => cancelAnimationFrame(firstFrame));

  const manifesto = $("#manifesto");
  const originalManifesto = manifesto?.textContent ?? "";
  const highlightedWords = ["photographed", "directly", "non-invasively.", "classical", "learned"];
  if (manifesto) {
    manifesto.innerHTML = originalManifesto
      .split(" ")
      .map((word) => `<span class="w${highlightedWords.includes(word) ? " hl" : ""}">${word}</span>`)
      .join(" ");
  }
  const words = manifesto ? $$(".w", manifesto) : [];

  const nav = $(".nav");
  const chapters = $$(".chapter");
  const links = $$(".nav a");
  const setTheme = (chapter) => {
    if (!nav || !chapter) return;
    nav.style.setProperty("--nav-bg", chapter.dataset.navBg);
    nav.style.setProperty("--nav-bg-solid", chapter.dataset.navSolid);
    nav.style.setProperty("--nav-ink", chapter.dataset.navInk);
    links.forEach((link) => {
      link.setAttribute("aria-current", link.dataset.target === root.dataset.view ? "true" : "false");
    });
  };

  const speedElements = $$ ("#overview [data-speed], #method [data-speed]");
  const timeline = $("#timeline");
  const steps = timeline ? $$(".step", timeline) : [];
  const setTimeline = (progress) => {
    if (!timeline) return;
    timeline.style.setProperty("--p", progress.toFixed(3));
    steps.forEach((step, index) => {
      step.classList.toggle("on", progress >= index / (steps.length - 1) - 0.001);
    });
  };

  const onScroll = () => {
    const viewportHeight = innerHeight;
    if (!reduceMotion) {
      speedElements.forEach((element) => {
        const rect = element.getBoundingClientRect();
        const offset = (rect.top + rect.height / 2 - viewportHeight / 2)
          * parseFloat(element.dataset.speed);
        element.style.translate = `0 ${offset.toFixed(1)}px`;
      });
      if (manifesto) {
        const rect = manifesto.getBoundingClientRect();
        const progress = Math.min(1, Math.max(0,
          (viewportHeight * 0.85 - rect.top) / (rect.height + viewportHeight * 0.3)));
        const visibleCount = Math.round(progress * words.length);
        words.forEach((word, index) => word.classList.toggle("on", index < visibleCount));
      }
      if (timeline) {
        const rect = timeline.getBoundingClientRect();
        setTimeline(Math.min(1, Math.max(0,
          (viewportHeight * 0.85 - rect.top) / (viewportHeight * 0.55))));
      }
    } else {
      setTimeline(1);
    }

    let currentChapter = chapters.find((chapter) => chapter.getClientRects().length) || chapters[0];
    chapters.forEach((chapter) => {
      if (chapter.getClientRects().length && chapter.getBoundingClientRect().top <= 80) currentChapter = chapter;
    });
    setTheme(currentChapter);
  };
  const onScrollEvent = () => {
    if (!scrollFrame) scrollFrame = requestAnimationFrame(() => {
      scrollFrame = 0;
      onScroll();
    });
  };
  listen(window, "scroll", onScrollEvent, { passive: true });
  listen(window, "resize", onScroll);
  onScroll();

  const wipe = $(".wipe");
  $$ ("[data-target]").forEach((link) => listen(link, "click", (event) => {
    const view = link.dataset.target;
    const target = document.getElementById(view === "home" ? "overview" : view);
    if (!target) return;
    event.preventDefault();
    wipe?.style.setProperty("--wipe", target.dataset.wipe || "#172B50");
    wipe?.classList.remove("go");
    if (wipe) void wipe.offsetWidth;
    wipe?.classList.add("go");
    actions.onNavigate?.(view);
    later(onScroll, reduceMotion ? 0 : 420);
  }));

  if (!reduceMotion && matchMedia("(hover: hover)").matches) {
    $$ ("[data-tilt]").forEach((element) => {
      const maxTilt = parseFloat(element.dataset.tiltMax || "4");
      listen(element, "pointermove", (event) => {
        const rect = element.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width - 0.5;
        const y = (event.clientY - rect.top) / rect.height - 0.5;
        element.style.transform = `perspective(1000px) rotateX(${(-y * maxTilt).toFixed(2)}deg) rotateY(${(x * maxTilt).toFixed(2)}deg)`;
      });
      listen(element, "pointerleave", () => { element.style.transform = ""; });
    });
  }

  const beforeAfter = $("#ba");
  const range = beforeAfter && $("input", beforeAfter);
  if (range) listen(range, "input", (event) => {
    beforeAfter.style.setProperty("--pos", `${event.target.value}%`);
  });

  const runButton = $("#runBtn");
  const uploadInput = $("#uploadInput");
  const resetSessionButton = $("#resetSessionBtn");
  if (resetSessionButton) listen(resetSessionButton, "click", () => actions.onReset?.());
  if (uploadInput) listen(uploadInput, "change", (event) => {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    actions.onFileSelected?.(file);
  });
  $$ ("input[name='model']").forEach((input) => listen(input, "change", (event) => {
    actions.onMethodChange?.(event.currentTarget.value);
  }));
  if (runButton) listen(runButton, "click", () => {
    actions.onRun?.($("input[name='model']:checked")?.value || "unet");
  });

  const algorithmViewers = $$("[data-algorithm-viewer]");
  const backgroundForMode = (viewer, mode) => {
    const originalUrl = $("[data-original-layer]", viewer)?.src;
    const maskUrl = $("[data-mask-layer]", viewer)?.src;
    if (mode === "mask") return `url("${maskUrl}")`;
    return `url("${maskUrl}"), url("${originalUrl}")`;
  };
  const setViewerMode = (viewer, mode) => {
    const loupe = $(".loupe", viewer);
    if (!viewer || !["mask", "overlay"].includes(mode)) return;
    viewer.dataset.mode = mode;
    $$(".tabs button", viewer).forEach((button) => {
      button.setAttribute("aria-selected", button.dataset.mode === mode ? "true" : "false");
    });
    if (loupe) {
      loupe.style.backgroundImage = backgroundForMode(viewer, mode);
      loupe.style.backgroundBlendMode = mode === "overlay" ? "screen, normal" : "normal";
    }
  };
  algorithmViewers.forEach((viewer) => {
    const tabs = $$(".tabs button", viewer);
    tabs.forEach((button) => {
      listen(button, "click", () => setViewerMode(viewer, button.dataset.mode));
      listen(button, "keydown", (event) => {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        const availableTabs = tabs.filter((tab) => !tab.disabled);
        if (availableTabs.length < 2) return;
        const direction = event.key === "ArrowRight" ? 1 : -1;
        const currentIndex = availableTabs.indexOf(button);
        const next = availableTabs[(currentIndex + direction + availableTabs.length) % availableTabs.length];
        next.focus();
        setViewerMode(viewer, next.dataset.mode);
      });
    });
    setViewerMode(viewer, viewer.dataset.mode || "mask");

    const loupe = $(".loupe", viewer);
    if (!loupe) return;
    listen(viewer, "pointermove", (event) => {
      if (event.target.closest(".tabs")) {
        loupe.style.opacity = 0;
        return;
      }
      loupe.style.opacity = "";
      const rect = viewer.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const source = $("[data-original-layer]", viewer);
      const sourceWidth = source?.naturalWidth || rect.width;
      const sourceHeight = source?.naturalHeight || rect.height;
      const scale = Math.min(rect.width / sourceWidth, rect.height / sourceHeight);
      const imageWidth = sourceWidth * scale;
      const imageHeight = sourceHeight * scale;
      const offsetX = (rect.width - imageWidth) / 2;
      const offsetY = (rect.height - imageHeight) / 2;
      const zoom = 2.6;
      const halfLoupe = loupe.offsetWidth / 2;
      loupe.style.left = `${x}px`;
      loupe.style.top = `${y}px`;
      loupe.style.backgroundSize = `${imageWidth * zoom}px ${imageHeight * zoom}px`;
      loupe.style.backgroundPosition = `${halfLoupe - offsetX - (x - offsetX) * zoom + x}px ${halfLoupe - offsetY - (y - offsetY) * zoom + y}px`;
    });
  });
  listen(root, "visionquest:results-updated", () => {
    algorithmViewers.forEach((viewer) => setViewerMode(viewer, viewer.dataset.mode || "mask"));
  });

  return () => {
    cleanups.forEach((cleanup) => cleanup());
    timers.forEach(clearTimeout);
    timers.clear();
    if (scrollFrame) cancelAnimationFrame(scrollFrame);
    if (manifesto) manifesto.textContent = originalManifesto;
  };
}
