import { marked } from 'marked';
import { gfm } from 'marked-gfm-heading-id';

export class MarkdownProcessor {
    constructor() {
        // Configure marked with GFM and other options
        marked.use(gfm());
        marked.setOptions({
            headerIds: true,
            gfm: true,
            breaks: true,
            pedantic: false,
            smartLists: true,
            smartypants: true
        });

        // Custom renderer for research paper formatting
        const renderer = new marked.Renderer();
        
        // Customize heading rendering
        renderer.heading = (text, level) => {
            const fontSize = this.getHeadingFontSize(level);
            return `<h${level} style="font-size: ${fontSize}pt; font-weight: bold; margin-top: 20px; margin-bottom: 10px;">${text}</h${level}>`;
        };

        // Customize paragraph rendering
        renderer.paragraph = (text) => {
            return `<p style="text-align: justify; line-height: 2.0;">${text}</p>`;
        };

        // Customize list rendering
        renderer.list = (body, ordered) => {
            const type = ordered ? 'ol' : 'ul';
            return `<${type} style="margin-left: 20px; line-height: 1.6;">${body}</${type}>`;
        };

        marked.use({ renderer });
    }

    getHeadingFontSize(level) {
        const sizes = {
            1: 16, // Title
            2: 14, // Section
            3: 13, // Subsection
            4: 12, // Sub-subsection
            5: 12,
            6: 12
        };
        return sizes[level] || 12;
    }

    process(markdown) {
        try {
            return marked(markdown);
        } catch (error) {
            console.error('Error processing markdown:', error);
            throw new Error('Failed to process markdown: ' + error.message);
        }
    }

    // Process specific sections of the research paper
    processAbstract(abstract) {
        return `<div style="margin-bottom: 20px;">
            <h2 style="font-size: 14pt; font-weight: bold;">Abstract</h2>
            ${this.process(abstract)}
        </div>`;
    }

    processKeywords(keywords) {
        const keywordList = Array.isArray(keywords) ? keywords : keywords.split(',').map(k => k.trim());
        return `<div style="margin-bottom: 20px;">
            <strong>Keywords:</strong> ${keywordList.join(', ')}
        </div>`;
    }

    // Helper method to clean up markdown text
    cleanMarkdown(text) {
        return text
            .replace(/\r\n/g, '\n')
            .replace(/\n{3,}/g, '\n\n')
            .trim();
    }
}
