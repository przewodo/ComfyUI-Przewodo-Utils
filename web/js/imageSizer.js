import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "przewodo.ImageSizer.DisplayDimensions",
    beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass === "przewodo ImageSizer") {
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDrawForeground?.apply?.(this, arguments);

                if (!this.flags.collapsed) {
                    // Try to get UI data from execution outputs
                    const v = app.nodeOutputs?.[this.id + ""];
                    
                    let dimensions = null;
                    
                    // Check for UI data from execution
                    if (v && v.dimensions) {
                        dimensions = v.dimensions[0];
                    } else {
                        dimensions = null;
                    }
                    
                    let text;
                    if (dimensions) {
                        text = `${dimensions.width} × ${dimensions.height}`;
                    } else {
                        text = `No dimensions available`;
                    }
                    
                    // Set font style
                    ctx.save();
                    ctx.font = "bold 12px sans-serif";
                    ctx.fillStyle = "#0055ff";
                    
                    // Measure text size
                    const sz = ctx.measureText(text);
                    const textWidth = sz.width;
                    const textHeight = 0; // Approximate height for 12px font
                    const padding = 3;
                    
                    // Calculate required width for the text to fit
                    const requiredWidth = textWidth + padding * 2;
                    const minNodeWidth = 150; // Minimum node width
                    const finalWidth = Math.max(minNodeWidth, requiredWidth);
                    
                    // Adjust node size if needed
                    if (this.size[0] < finalWidth) {
                        this.size[0] = finalWidth;
                        this.setDirtyCanvas(true, true); // Force full redraw
                    }
                    
                    // Ensure minimum height to show the text
                    const baseHeight = this.computeSize()[1];
                    const minHeight = baseHeight + textHeight;
                    if (this.size[1] < minHeight) {
                        this.size[1] = minHeight;
                        this.setDirtyCanvas(true, true);
                    }
                    
                    // Draw the text at the bottom of the node
                    ctx.fillText(text, this.size[0] - textWidth - 5, this.size[1] - 5);
                    ctx.restore();
                }

                return r;
            };

            // Also hook into node input changes to update display immediately
            const onWidget = nodeType.prototype.onWidget;
            nodeType.prototype.onWidget = function(e, widget, ...args) {
                const result = onWidget ? onWidget.apply(this, [e, widget, ...args]) : undefined;
                
                // Trigger redraw when input values change
                if (widget.name === "model_type" || 
                    widget.name === "aspect_ratio_width" || 
                    widget.name === "aspect_ratio_height" || 
                    widget.name === "image_scale") {
                    this.setDirtyCanvas(true, true); // Force full redraw to recalculate size
                }
                
                return result;
            };

            // Override computeSize to ensure minimum size for text
            const originalComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function(out) {
                const size = originalComputeSize ? originalComputeSize.call(this, out) : [150, 100];
                
                // Add extra space for dimension text
                const extraHeight = 20; // Space for dimension text
                size[1] = Math.max(size[1], size[1] + extraHeight);
                
                return size;
            };
        }
    },
});
