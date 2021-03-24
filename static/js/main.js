const Top = document.querySelector(".to-top");

window.addEventListener("scroll", () => {
  if (window.pageYOffset > 200) {
    Top.classList.add("active");
  } else {
    Top.classList.remove("active");
  }
});

function changemode(themeObj) {
  if (themeObj.dataset.mode == "light") {
    document.documentElement.setAttribute("data-theme", "dark");
    themeObj.dataset.mode = "dark";
  } else {
    document.documentElement.setAttribute("data-theme", "light");
    themeObj.dataset.mode = "light";
  }
}