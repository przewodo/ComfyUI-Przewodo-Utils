import { app } from "../../scripts/app.js";

// Function to load CSS
function loadCSS(href) {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.type = 'text/css';
    link.href = href;
    document.head.appendChild(link);
    console.log(`[WAN Node Styling] Loaded CSS: ${href}`);
}

// Function to apply additional dynamic styling
function applyDynamicStyling() {
    // Additional JavaScript-based styling can go here
    const style = document.createElement('style');
    style.textContent = `
        /* Additional dynamic styles if needed */
        .comfy-graph-canvas .litegraph .node[title*="przewodo WanImageToVideoAdvancedSampler"] {
            /* Ensure our custom styles take precedence */
            position: relative;
        }
    `;
    document.head.appendChild(style);
}

// Register the extension
app.registerExtension({
    name: "Przewodo.WanNodeStyling",
    
    async setup() {
        console.log("[WAN Node Styling] Extension loading...");
        
        // Load the CSS file
        const cssPath = `extensions/ComfyUI-Przewodo-Utils/wan_node_styling.css`;
        loadCSS(cssPath);
        
        // Apply any additional dynamic styling
        applyDynamicStyling();
        
        console.log("[WAN Node Styling] Extension loaded successfully");
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Hook into node creation to apply custom styling
        if (nodeData.name === "przewodo WanImageToVideoAdvancedSampler") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const result = originalOnNodeCreated?.apply(this, arguments);
                
                // Add custom class for easier targeting
                this.addProperty?.("wan_custom_styled", true);
                
                // Force a visual update
                setTimeout(() => {
                    this.setDirtyCanvas?.(true, true);
                }, 100);
                
                console.log("[WAN Node Styling] Applied custom styling to node:", this.title);
                return result;
            };
        }
    }
});

console.log("[WAN Node Styling] Extension script loaded");
