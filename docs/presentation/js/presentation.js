(function () {
  const SPEAKER_KEY = "vmbench-speaker-state";
  const COMMAND_KEY = "vmbench-speaker-command";
  let speakerWindow = null;

  function getSlides() {
    return Array.from(document.querySelectorAll(".slides > section"));
  }

  function getSlideTitle(slide) {
    if (!slide) return "";
    const heading = slide.querySelector("h1, h2, h3");
    return heading ? heading.textContent.trim() : "Untitled slide";
  }

  function getSlideNotes(slide) {
    if (!slide) return "";
    const notes = slide.querySelector("aside.notes");
    return notes ? notes.innerHTML.trim() : "";
  }

  function updateSpeakerState() {
    if (typeof Reveal === "undefined") return;
    const slides = getSlides();
    const indices = Reveal.getIndices();
    const currentSlide = Reveal.getCurrentSlide();
    const nextSlide = slides[indices.h + 1] || null;
    const state = {
      currentTitle: getSlideTitle(currentSlide),
      nextTitle: getSlideTitle(nextSlide),
      notesHtml: getSlideNotes(currentSlide),
      currentIndex: indices.h + 1,
      total: slides.length
    };

    try {
      window.localStorage.setItem(SPEAKER_KEY, JSON.stringify(state));
    } catch (error) {
      // Ignore if localStorage is unavailable.
    }
  }

  function openSpeakerWindow() {
    const url = new URL("speaker.html", window.location.href).toString();
    if (!speakerWindow || speakerWindow.closed) {
      speakerWindow = window.open(url, "VmbenchSpeaker", "popup=yes,width=1100,height=900");
    } else {
      speakerWindow.focus();
    }
    updateSpeakerState();
  }

  function handleSpeakerCommand(command) {
    if (typeof Reveal === "undefined") return;
    if (command === "next") Reveal.next();
    if (command === "prev") Reveal.prev();
  }

  function activatePipeline() {
    const slide = Reveal.getCurrentSlide();
    if (!slide) return;

    const stages = slide.querySelectorAll("[data-stage]");
    if (!stages.length) return;

    let active = 0;
    function render() {
      stages.forEach((el, index) => {
        el.classList.toggle("active", index <= active);
      });
    }

    render();
    const timer = setInterval(() => {
      active += 1;
      if (active >= stages.length) {
        clearInterval(timer);
        return;
      }
      render();
    }, 900);
    slide._stageTimer = timer;
  }

  function clearTimer(slide) {
    if (slide && slide._stageTimer) {
      clearInterval(slide._stageTimer);
      slide._stageTimer = null;
    }
  }

  Reveal.on("ready", activatePipeline);
  Reveal.on("ready", updateSpeakerState);
  Reveal.on("slidechanged", (event) => {
    clearTimer(event.previousSlide);
    activatePipeline();
    updateSpeakerState();
  });

  window.addEventListener("keydown", (event) => {
    if ((event.key === "s" || event.key === "S") && !event.metaKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault();
      event.stopPropagation();
      openSpeakerWindow();
    }
  }, true);

  window.addEventListener("storage", (event) => {
    if (event.key !== COMMAND_KEY || !event.newValue) return;
    handleSpeakerCommand(event.newValue);
    try {
      window.localStorage.removeItem(COMMAND_KEY);
    } catch (error) {
      // Ignore cleanup issues.
    }
  });
})();
