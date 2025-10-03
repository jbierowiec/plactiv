// static/js/speed.js
document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("flyover"); // may be null on 3D page
  const buttons = document.querySelectorAll(".speed-btn");

  function setActive(btn) {
    buttons.forEach(b => b.classList.remove("active", "btn-primary"));
    buttons.forEach(b => b.classList.add("btn-outline-primary"));
    btn.classList.remove("btn-outline-primary");
    btn.classList.add("btn-primary", "active");
  }

  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      setActive(btn);
      if (video) {
        const rate = parseFloat(btn.getAttribute("data-rate"));
        video.playbackRate = rate;
        if (video.paused) video.play();
      }
      // On the 3D map page, map3d.js reads the active button to set speed.
    });
  });
});
