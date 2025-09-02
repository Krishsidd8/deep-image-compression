function toggleMode() {
  document.body.classList.toggle("dark");

  const button = document.querySelector(".toggle");
  if (document.body.classList.contains("dark")) {
    button.textContent = "☀️ Light Mode";
  } else {
    button.textContent = "🌙 Dark Mode";
  }
}

let currentSlide = 0;
const slides = document.querySelectorAll('.carousel-slide');

function showSlide(index) {
  slides.forEach((slide, i) => {
    slide.classList.toggle('active', i === index);
  });
}

function changeSlide(direction) {
  currentSlide = (currentSlide + direction + slides.length) % slides.length;
  showSlide(currentSlide);
}

// Initialize first slide
document.addEventListener("DOMContentLoaded", () => showSlide(currentSlide));
