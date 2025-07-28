// Simple CSS injection for WAN node styling
(function() {
    'use strict';
    
    // CSS styles for the WAN node
    const wanNodeCSS = `
        /* WAN Image to Video Advanced Sampler Custom Styling */
        .comfy-graph-canvas .litegraph .node {
            /* Check if this is our target node */
        }
        
        /* Target nodes by title attribute */
        .litegraph .lnode[title*="WanImageToVideoAdvancedSampler"],
        .litegraph .lnode[title*="przewodo WanImageToVideoAdvancedSampler"] {
            background-color: #2C5282 !important;
            border: 2px solid #DA691E !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 8px rgba(74, 144, 226, 0.3) !important;
        }
        
        /* Style the title */
        .litegraph .lnode[title*="WanImageToVideoAdvancedSampler"] .node_title,
        .litegraph .lnode[title*="przewodo WanImageToVideoAdvancedSampler"] .node_title {
            background: linear-gradient(135deg, #4A90E2, #357ABD) !important;
            color: #FFFFFF !important;
            font-weight: bold !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5) !important;
            border-radius: 6px 6px 0 0 !important;
        }
        
        /* Selected state */
        .litegraph .lnode[title*="WanImageToVideoAdvancedSampler"].selected,
        .litegraph .lnode[title*="przewodo WanImageToVideoAdvancedSampler"].selected {
            border-color: #FFD700 !important;
            box-shadow: 0 0 15px rgba(255, 215, 0, 0.6) !important;
        }
        
        /* Executing state with animation */
        .litegraph .lnode[title*="WanImageToVideoAdvancedSampler"].executing,
        .litegraph .lnode[title*="przewodo WanImageToVideoAdvancedSampler"].executing {
            border-color: #00FF00 !important;
            animation: wan-execute-pulse 2s infinite;
        }
        
        @keyframes wan-execute-pulse {
            0%, 100% { box-shadow: 0 0 5px rgba(0, 255, 0, 0.5); }
            50% { box-shadow: 0 0 20px rgba(0, 255, 0, 0.8); }
        }
        
        /* Widgets styling */
        .litegraph .lnode[title*="WanImageToVideoAdvancedSampler"] .widget,
        .litegraph .lnode[title*="przewodo WanImageToVideoAdvancedSampler"] .widget {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 4px !important;
        }
        
        /* Alternative targeting by category */
        .litegraph .lnode[data-category*="PrzewodoUtils/Wan"] {
            border-left: 4px solid #4A90E2 !important;
        }
    `;
    
    // Function to inject CSS
    function injectCSS() {
        const style = document.createElement('style');
        style.type = 'text/css';
        style.id = 'wan-node-custom-styling';
        style.textContent = wanNodeCSS;
        document.head.appendChild(style);
        console.log('[WAN Styling] Custom CSS injected successfully');
    }
    
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectCSS);
    } else {
        injectCSS();
    }
    
    // Also try to inject after a delay to ensure ComfyUI is loaded
    setTimeout(injectCSS, 1000);
    
})();
