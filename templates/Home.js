// script.js
document.getElementById('getStartedBtn').addEventListener('click', function() {
    // Redirect to the recommendations page or perform other actions
    // For example:
    window.location.href = 'recommendations.html'; // Replace with your actual URL
    // Or you can use a more complex logic here, like showing a form, etc.
});

// Smooth scrolling for navigation links (optional)
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();

        const targetId = this.getAttribute('href').substring(1); // Get the target element ID
        const targetElement = document.getElementById(targetId);

        if (targetElement) {
            window.scrollTo({
                top: targetElement.offsetTop,
                behavior: 'smooth'
            });
        }
    });
});

// Image Slider (Example - You'll need to adapt it to your specific needs)
const heroImage = document.querySelector('.hero-image img');
const imageURLs = ['crop4.jpg', 'crop5.jpg', 'crop7.jpg']; // Array of image URLs
let currentImageIndex = 0;

setInterval(() => {
    heroImage.src = imageURLs[currentImageIndex];
    currentImageIndex = (currentImageIndex + 1) % imageURLs.length; // Cycle through images
}, 3000); // Change image every 3 seconds