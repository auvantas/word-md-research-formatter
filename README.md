# Research Paper Formatter Word Add-in

This Microsoft Word add-in allows you to paste Markdown text and automatically format it into a research paper, including automatic citation formatting in various academic styles (APA, Harvard, MLA, Chicago).

## Features

- Convert Markdown text to properly formatted Word documents
- Automatic citation detection and formatting
- Support for multiple citation styles:
  - APA
  - Harvard
  - MLA
  - Chicago
- Maintains proper research paper structure

## Prerequisites

- Microsoft Word (2016 or later)
- Node.js and npm (for development)

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```

## Usage

1. Open Microsoft Word
2. Go to the Home tab
3. Click on the "Format Research Paper" button in the ribbon
4. Paste your Markdown text into the input area
5. Select your desired citation style
6. Click "Format Paper"

## Development

This project uses:
- Office.js for Word integration
- Marked.js for Markdown parsing
- Citation.js for citation formatting
- Office UI Fabric React for the user interface

## Project Structure

```
word-md-research-formatter/
├── src/
│   └── taskpane/
│       ├── taskpane.html
│       ├── taskpane.css
│       └── taskpane.js
├── manifest.xml
├── package.json
└── README.md
```

## License

MIT
