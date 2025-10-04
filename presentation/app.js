// Slide navigation functionality
let currentSlideIndex = 1;
const totalSlides = 7;

// Initialize the presentation
document.addEventListener('DOMContentLoaded', function() {
    updateSlideDisplay();
    setupKeyboardNavigation();
    setupTouchNavigation();
});

// Change slide by offset (next/previous)
function changeSlide(direction) {
    const newIndex = currentSlideIndex + direction;
    
    if (newIndex >= 1 && newIndex <= totalSlides) {
        currentSlideIndex = newIndex;
        updateSlideDisplay();
    }
}

// Go to specific slide
function currentSlide(slideNumber) {
    if (slideNumber >= 1 && slideNumber <= totalSlides) {
        currentSlideIndex = slideNumber;
        updateSlideDisplay();
    }
}

// Update the slide display and navigation
function updateSlideDisplay() {
    // Hide all slides
    const slides = document.querySelectorAll('.slide');
    slides.forEach(slide => {
        slide.classList.remove('active');
    });
    
    // Show current slide
    const currentSlideElement = document.getElementById(`slide-${currentSlideIndex}`);
    if (currentSlideElement) {
        currentSlideElement.classList.add('active');
    }
    
    // Update slide counter
    const slideCounter = document.getElementById('currentSlide');
    if (slideCounter) {
        slideCounter.textContent = currentSlideIndex;
    }
    
    // Update navigation buttons
    updateNavigationButtons();
    
    // Update dots
    updateDots();
    
    // Add slide transition animation
    addSlideAnimation();
}

// Update navigation button states
function updateNavigationButtons() {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    
    if (prevBtn) {
        prevBtn.disabled = currentSlideIndex === 1;
    }
    
    if (nextBtn) {
        nextBtn.disabled = currentSlideIndex === totalSlides;
    }
}

// Update dot indicators
function updateDots() {
    const dots = document.querySelectorAll('.dot');
    dots.forEach((dot, index) => {
        if (index + 1 === currentSlideIndex) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    });
}

// Add slide transition animation
function addSlideAnimation() {
    const currentSlideElement = document.getElementById(`slide-${currentSlideIndex}`);
    if (currentSlideElement) {
        // Remove any existing animation classes
        currentSlideElement.classList.remove('slide-in-left', 'slide-in-right');
        
        // Force reflow
        void currentSlideElement.offsetWidth;
        
        // Add appropriate animation class
        currentSlideElement.classList.add('slide-in-right');
    }
}

// Setup keyboard navigation
function setupKeyboardNavigation() {
    document.addEventListener('keydown', function(event) {
        switch(event.key) {
            case 'ArrowLeft':
            case 'ArrowUp':
                changeSlide(-1);
                event.preventDefault();
                break;
            case 'ArrowRight':
            case 'ArrowDown':
            case ' ': // Space key
                changeSlide(1);
                event.preventDefault();
                break;
            case 'Home':
                currentSlide(1);
                event.preventDefault();
                break;
            case 'End':
                currentSlide(totalSlides);
                event.preventDefault();
                break;
            case 'Escape':
                // Optional: Exit fullscreen or reset
                break;
        }
    });
}

// Setup touch navigation for mobile devices
function setupTouchNavigation() {
    let startX = 0;
    let startY = 0;
    const threshold = 50; // Minimum distance for swipe
    
    document.addEventListener('touchstart', function(event) {
        startX = event.touches[0].clientX;
        startY = event.touches[0].clientY;
    });
    
    document.addEventListener('touchend', function(event) {
        if (!startX || !startY) {
            return;
        }
        
        const endX = event.changedTouches[0].clientX;
        const endY = event.changedTouches[0].clientY;
        
        const deltaX = startX - endX;
        const deltaY = startY - endY;
        
        // Check if horizontal swipe is greater than vertical
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
            if (Math.abs(deltaX) > threshold) {
                if (deltaX > 0) {
                    // Swipe left - next slide
                    changeSlide(1);
                } else {
                    // Swipe right - previous slide
                    changeSlide(-1);
                }
            }
        }
        
        // Reset start coordinates
        startX = 0;
        startY = 0;
    });
}

// Auto-advance slides (optional feature)
let autoAdvanceTimer = null;
const autoAdvanceDelay = 30000; // 30 seconds

function startAutoAdvance() {
    stopAutoAdvance();
    autoAdvanceTimer = setInterval(function() {
        if (currentSlideIndex < totalSlides) {
            changeSlide(1);
        } else {
            // Loop back to first slide or stop
            currentSlide(1);
        }
    }, autoAdvanceDelay);
}

function stopAutoAdvance() {
    if (autoAdvanceTimer) {
        clearInterval(autoAdvanceTimer);
        autoAdvanceTimer = null;
    }
}

// Presentation controls
function toggleAutoAdvance() {
    if (autoAdvanceTimer) {
        stopAutoAdvance();
    } else {
        startAutoAdvance();
    }
}

function enterFullscreen() {
    const element = document.documentElement;
    if (element.requestFullscreen) {
        element.requestFullscreen();
    } else if (element.mozRequestFullScreen) {
        element.mozRequestFullScreen();
    } else if (element.webkitRequestFullscreen) {
        element.webkitRequestFullscreen();
    } else if (element.msRequestFullscreen) {
        element.msRequestFullscreen();
    }
}

function exitFullscreen() {
    if (document.exitFullscreen) {
        document.exitFullscreen();
    } else if (document.mozCancelFullScreen) {
        document.mozCancelFullScreen();
    } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
    } else if (document.msExitFullscreen) {
        document.msExitFullscreen();
    }
}

// Progress tracking
function getProgress() {
    return {
        currentSlide: currentSlideIndex,
        totalSlides: totalSlides,
        percentage: Math.round((currentSlideIndex / totalSlides) * 100)
    };
}

// Analytics (optional)
function trackSlideView(slideNumber) {
    // Optional: Send analytics data
    console.log(`Slide ${slideNumber} viewed`);
}

// Initialize analytics tracking
document.addEventListener('DOMContentLoaded', function() {
    // Track initial slide view
    trackSlideView(currentSlideIndex);
    
    // Track slide changes
    const originalChangeSlide = changeSlide;
    changeSlide = function(direction) {
        originalChangeSlide(direction);
        trackSlideView(currentSlideIndex);
    };
    
    const originalCurrentSlide = currentSlide;
    currentSlide = function(slideNumber) {
        originalCurrentSlide(slideNumber);
        trackSlideView(currentSlideIndex);
    };
});

// Print support
function printSlides() {
    // Show all slides for printing
    const slides = document.querySelectorAll('.slide');
    slides.forEach(slide => {
        slide.style.position = 'relative';
        slide.style.opacity = '1';
        slide.style.transform = 'none';
        slide.style.pageBreakAfter = 'always';
    });
    
    // Hide navigation
    const navControls = document.querySelector('.nav-controls');
    const slideIndicators = document.querySelector('.slide-indicators');
    
    if (navControls) navControls.style.display = 'none';
    if (slideIndicators) slideIndicators.style.display = 'none';
    
    // Print
    window.print();
    
    // Restore normal view after printing
    setTimeout(function() {
        location.reload();
    }, 1000);
}

// Responsive handling
function handleResize() {
    // Adjust layout based on screen size
    const isMobile = window.innerWidth <= 768;
    const slides = document.querySelectorAll('.slide-content');
    
    slides.forEach(slide => {
        if (isMobile) {
            slide.style.padding = '1rem';
        } else {
            slide.style.padding = '2rem';
        }
    });
}

window.addEventListener('resize', handleResize);
window.addEventListener('orientationchange', function() {
    setTimeout(handleResize, 100);
});

// Initialize
handleResize();

// Export functions for external use
window.presentationAPI = {
    changeSlide: changeSlide,
    currentSlide: currentSlide,
    getProgress: getProgress,
    startAutoAdvance: startAutoAdvance,
    stopAutoAdvance: stopAutoAdvance,
    enterFullscreen: enterFullscreen,
    exitFullscreen: exitFullscreen,
    printSlides: printSlides
};

// Service Worker registration for offline support (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed');
            });
    });
}