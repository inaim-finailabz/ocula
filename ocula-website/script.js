// Model Data
const models = [
    {
        id: 1,
        name: 'SmolVLM-256M',
        tier: 'free',
        parameters: '256M',
        size: '150 MB',
        features: [
            'Instant object identification',
            'Basic image recognition',
            'Fast inference',
            'Low memory usage'
        ],
        downloadUrl: '#',
        description: 'Lightning-fast model for quick object recognition tasks.'
    },
    {
        id: 2,
        name: 'Moondream 3',
        tier: 'plus',
        parameters: '0.5B',
        size: '300 MB',
        features: [
            'Advanced object detection',
            'Object counting',
            'Receipt analysis',
            'Document understanding',
            'Enhanced accuracy'
        ],
        downloadUrl: '#',
        description: 'Balanced model for everyday vision tasks with improved accuracy.'
    },
    {
        id: 3,
        name: 'Qwen2-VL-2B',
        tier: 'pro',
        parameters: '2B',
        size: '1.2 GB',
        features: [
            'Complex reasoning',
            'Document OCR',
            'Multi-step analysis',
            'High accuracy',
            'Batch processing'
        ],
        downloadUrl: '#',
        description: 'Powerful model for complex visual reasoning tasks.'
    },
    {
        id: 4,
        name: 'LLaVA-v1.6-7B',
        tier: 'pro',
        parameters: '7B',
        size: '4.2 GB',
        features: [
            'State-of-the-art performance',
            'Advanced reasoning',
            'Detailed analysis',
            'Multi-modal understanding',
            'Best accuracy'
        ],
        downloadUrl: '#',
        description: 'Our most powerful model for the most demanding vision tasks.'
    },
    {
        id: 5,
        name: 'SmolVLM-500M',
        tier: 'plus',
        parameters: '500M',
        size: '280 MB',
        features: [
            'Quick object detection',
            'Scene understanding',
            'Text recognition',
            'Fast processing'
        ],
        downloadUrl: '#',
        description: 'Enhanced version of SmolVLM with better accuracy.'
    },
    {
        id: 6,
        name: 'Qwen2-VL-7B',
        tier: 'pro',
        parameters: '7B',
        size: '4.0 GB',
        features: [
            'Enterprise-grade performance',
            'Complex document analysis',
            'Advanced OCR',
            'Production ready'
        ],
        downloadUrl: '#',
        description: 'Enterprise model for professional applications.'
    }
];

// DOM Elements
const modelsGrid = document.getElementById('modelsGrid');
const loadingState = document.getElementById('loadingState');
const errorState = document.getElementById('errorState');
const filterButtons = document.querySelectorAll('.filter-btn');
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const downloadModal = document.getElementById('downloadModal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');

// Initialize
let currentFilter = 'all';

// Load Models
function loadModels() {
    // Show loading state
    loadingState.style.display = 'block';
    errorState.style.display = 'none';
    modelsGrid.innerHTML = '';
    
    // Simulate API delay
    setTimeout(() => {
        try {
            renderModels(models);
            loadingState.style.display = 'none';
        } catch (error) {
            loadingState.style.display = 'none';
            errorState.style.display = 'block';
        }
    }, 800);
}

// Render Models
function renderModels(modelsToRender) {
    const filteredModels = currentFilter === 'all' 
        ? modelsToRender 
        : modelsToRender.filter(model => model.tier === currentFilter);
    
    if (filteredModels.length === 0) {
        modelsGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 3rem;">
                <i class="fas fa-search" style="font-size: 3rem; color: var(--gray); margin-bottom: 1rem;"></i>
                <h3 style="color: var(--dark); margin-bottom: 0.5rem;">No models found</h3>
                <p style="color: var(--gray);">Try selecting a different filter</p>
            </div>
        `;
        return;
    }
    
    modelsGrid.innerHTML = filteredModels.map(model => `
        <div class="model-card" data-tier="${model.tier}">
            <div class="model-header">
                <div class="model-name">${model.name}</div>
                <span class="model-tier tier-${model.tier}">${model.tier.charAt(0).toUpperCase() + model.tier.slice(1)}</span>
            </div>
            <div class="model-body">
                <div class="model-specs">
                    <div class="spec-item">
                        <span class="spec-label">Parameters</span>
                        <span class="spec-value">${model.parameters}</span>
                    </div>
                    <div class="spec-item">
                        <span class="spec-label">Size</span>
                        <span class="spec-value">${model.size}</span>
                    </div>
                </div>
                <ul class="model-features">
                    ${model.features.slice(0, 3).map(feature => 
                        `<li><i class="fas fa-check"></i> ${feature}</li>`
                    ).join('')}
                </ul>
                <div class="model-actions">
                    <button class="btn btn-outline" onclick="showModelDetails(${model.id})">
                        <i class="fas fa-info"></i> Details
                    </button>
                    <button class="btn btn-primary" onclick="downloadModel(${model.id})">
                        <i class="fas fa-download"></i> Download
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

// Filter Models
filterButtons.forEach(button => {
    button.addEventListener('click', () => {
        filterButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        currentFilter = button.dataset.filter;
        renderModels(models);
    });
});

// Show Model Details
function showModelDetails(modelId) {
    const model = models.find(m => m.id === modelId);
    if (!model) return;
    
    modalTitle.textContent = model.name;
    modalBody.innerHTML = `
        <div style="margin-bottom: 1.5rem;">
            <span class="model-tier tier-${model.tier}" style="margin-bottom: 1rem;">${model.tier.charAt(0).toUpperCase() + model.tier.slice(1)}</span>
            <p style="color: var(--gray); margin-top: 0.5rem;">${model.description}</p>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
            <div>
                <strong style="display: block; margin-bottom: 0.25rem;">Parameters</strong>
                <span style="color: var(--gray);">${model.parameters}</span>
            </div>
            <div>
                <strong style="display: block; margin-bottom: 0.25rem;">Size</strong>
                <span style="color: var(--gray);">${model.size}</span>
            </div>
        </div>
        
        <div style="margin-bottom: 1.5rem;">
            <strong style="display: block; margin-bottom: 0.5rem;">Features</strong>
            <ul style="list-style: none; padding: 0;">
                ${model.features.map(feature => 
                    `<li style="padding: 0.25rem 0; color: var(--gray);">
                        <i class="fas fa-check" style="color: var(--secondary); margin-right: 0.5rem;"></i>${feature}
                    </li>`
                ).join('')}
            </ul>
        </div>
        
        <button class="btn btn-primary" style="width: 100%;" onclick="downloadModel(${model.id}); closeModal();">
            <i class="fas fa-download"></i> Download Model
        </button>
    `;
    
    downloadModal.classList.add('active');
}

// Download Model
function downloadModel(modelId) {
    const model = models.find(m => m.id === modelId);
    if (!model) return;
    
    // Show toast notification
    showToast(`Starting download for ${model.name}...`);
    
    // Simulate download (in real implementation, this would trigger actual download)
    console.log(`Downloading ${model.name} from ${model.downloadUrl}`);
    
    // You could add actual download logic here:
    // window.location.href = model.downloadUrl;
    // or create an invisible link element to trigger download
}

// Close Modal
function closeModal() {
    downloadModal.classList.remove('active');
}

// Close modal when clicking outside
modalBody.addEventListener('click', (e) => {
    e.stopPropagation();
});

downloadModal.addEventListener('click', () => {
    closeModal();
});

// Show Toast Notification
function showToast(message) {
    toastMessage.textContent = message;
    toast.classList.add('active');
    
    setTimeout(() => {
        toast.classList.remove('active');
    }, 3000);
}

// Mobile Menu Toggle
mobileMenuBtn.addEventListener('click', () => {
    const navLinks = document.querySelector('.nav-links');
    navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
    
    if (navLinks.style.display === 'flex') {
        navLinks.style.flexDirection = 'column';
        navLinks.style.position = 'absolute';
        navLinks.style.top = '100%';
        navLinks.style.left = '0';
        navLinks.style.right = '0';
        navLinks.style.background = 'var(--dark)';
        navLinks.style.padding = '1rem';
        navLinks.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
    }
});

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
            
            // Close mobile menu if open
            const navLinks = document.querySelector('.nav-links');
            if (window.innerWidth <= 768) {
                navLinks.style.display = 'none';
            }
        }
    });
});

// Close mobile menu when clicking outside
document.addEventListener('click', (e) => {
    const navLinks = document.querySelector('.nav-links');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    
    if (!navLinks.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
        if (window.innerWidth <= 768) {
            navLinks.style.display = 'none';
        }
    }
});

// Handle window resize
window.addEventListener('resize', () => {
    const navLinks = document.querySelector('.nav-links');
    if (window.innerWidth > 768) {
        navLinks.style.display = 'flex';
        navLinks.style.flexDirection = 'row';
        navLinks.style.position = 'static';
        navLinks.style.boxShadow = 'none';
        navLinks.style.padding = '0';
    } else {
        navLinks.style.display = 'none';
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
});