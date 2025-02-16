export class TOCGenerator {
    constructor() {
        this.headings = [];
        this.maxLevel = 3; // Default max level for TOC
    }

    extractHeadings(text) {
        // Reset headings
        this.headings = [];
        
        // Match all markdown headings (# Heading)
        const headingRegex = /^(#{1,6})\s+(.+)$/gm;
        let match;
        
        while ((match = headingRegex.exec(text)) !== null) {
            const level = match[1].length;
            const title = match[2].trim();
            
            if (level <= this.maxLevel) {
                this.headings.push({
                    level,
                    title,
                    id: this.generateHeadingId(title)
                });
            }
        }
        
        return this.headings;
    }

    generateHeadingId(title) {
        // Convert title to URL-friendly ID
        return title
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/(^-|-$)/g, '');
    }

    async generateTOC(context) {
        if (this.headings.length === 0) {
            return;
        }

        try {
            await context.sync(async () => {
                // Insert "Table of Contents" heading
                const range = context.document.body.insertParagraph("Table of Contents", "Start");
                range.font.bold = true;
                range.font.size = 14;
                range.alignment = "center";
                
                // Add spacing after TOC heading
                const spacingPara = context.document.body.insertParagraph("", "After");
                spacingPara.font.size = 12;
                
                // Generate TOC entries
                for (const heading of this.headings) {
                    const indent = "  ".repeat(heading.level - 1);
                    const dots = ".".repeat(50 - indent.length - heading.title.length);
                    const entry = `${indent}${heading.title} ${dots}`;
                    
                    const tocEntry = context.document.body.insertParagraph(entry, "End");
                    tocEntry.font.name = "Times New Roman";
                    tocEntry.font.size = 12;
                    
                    // Add page number field
                    const pageNumber = tocEntry.insertField("Page", "End");
                    pageNumber.font.name = "Times New Roman";
                    pageNumber.font.size = 12;
                }
                
                // Add spacing after TOC
                const endSpacingPara = context.document.body.insertParagraph("", "End");
                endSpacingPara.font.size = 12;
            });
        } catch (error) {
            console.error("Error generating TOC:", error);
            throw error;
        }
    }

    async updatePageNumbers(context) {
        try {
            await context.sync(async () => {
                // Update all fields in the document
                context.document.fields.getItem().update();
            });
        } catch (error) {
            console.error("Error updating page numbers:", error);
            throw error;
        }
    }
}
