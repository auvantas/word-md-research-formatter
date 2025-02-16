// Import required libraries (these will be bundled by webpack)
import { CitationParser } from './citationParser';
import { MarkdownProcessor } from './markdownProcessor';
import { WordFormatter } from './wordFormatter';
import { paperFormats } from './paperFormats';
import { TOCGenerator } from './tocGenerator';

export default class App {
    constructor() {
        this.citationParser = new CitationParser();
        this.markdownProcessor = new MarkdownProcessor();
        this.wordFormatter = new WordFormatter();
        this.tocGenerator = new TOCGenerator();
    }

    async initialize() {
        Office.onReady((info) => {
            if (info.host === Office.HostType.Word) {
                document.getElementById("formatButton").onclick = this.processDocument.bind(this);
                this.setupErrorHandling();
                // Add format selection dropdown
                const formatSelect = document.getElementById('paper-format');
                Object.keys(paperFormats).forEach(format => {
                    const option = document.createElement('option');
                    option.value = format;
                    option.textContent = paperFormats[format].name;
                    formatSelect.appendChild(option);
                });
            }
        });
    }

    async processDocument() {
        try {
            await Word.run(async (context) => {
                this.updateStatus('Processing document...');
                
                // Get input values
                const markdownText = document.getElementById('markdown-input').value;
                const citationStyle = document.getElementById('citation-style').value;
                const paperFormat = document.getElementById('paper-format').value;
                const bibFile = document.getElementById('bib-file').files[0];
                
                // Process citations
                const citations = await this.citationParser.parseCitations(markdownText, bibFile);
                
                // Extract headings for TOC
                this.tocGenerator.extractHeadings(markdownText);
                
                // Convert markdown to HTML
                const html = await this.markdownProcessor.convertToHtml(markdownText, citations);
                
                // Clear document
                context.document.body.clear();
                
                // Generate TOC
                await this.tocGenerator.generateTOC(context);
                
                // Insert content
                const range = context.document.body.insertHtml(html, "End");
                
                // Apply paper format
                await this.wordFormatter.applyFormat(context, paperFormats[paperFormat]);
                
                // Format citations
                await this.formatCitations(context, citations, citationStyle);
                
                // Update page numbers in TOC
                await this.tocGenerator.updatePageNumbers(context);
                
                await context.sync();
                this.updateStatus('Document processed successfully!');
            });
        } catch (error) {
            console.error('Error processing document:', error);
            this.updateStatus('Error processing document: ' + error.message, true);
        }
    }

    setupErrorHandling() {
        window.onerror = function(message, source, lineno, colno, error) {
            this.showError(`Error: ${message}`);
            console.error('Error:', error);
            return false;
        }.bind(this);
    }

    formatReferences(citations, style) {
        let html = "";
        citations.forEach((citation, index) => {
            const formattedCitation = this.citationParser.formatCitation(citation, style);
            html += `<p style="text-indent: -36px; padding-left: 36px; margin-bottom: 12px;">
                ${index + 1}. ${formattedCitation}
            </p>`;
        });
        return html;
    }

    async formatCitations(context, citations, style) {
        const referencesHtml = this.formatReferences(citations, style);
        await this.wordFormatter.insertReferences(context, referencesHtml);
    }

    showError(message) {
        const statusDiv = document.getElementById("status");
        statusDiv.innerHTML = `<div style="color: red; padding: 10px;">${message}</div>`;
    }

    updateStatus(message, error = false) {
        const statusDiv = document.getElementById("status");
        statusDiv.innerHTML = `<div style="color: ${error ? 'red' : '#666'}; padding: 10px;">${message}</div>`;
    }
}
