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

### Markdown Support

The add-in supports standard Markdown syntax including:
- Headers (# to ######)
- Bold and italic text (**bold**, *italic*)
- Lists (ordered and unordered)
- Code blocks (inline and fenced)
- Block quotes
- Tables
- Links and images

### Citation Format

Citations can be included in your Markdown using the following formats:
- Inline citation: [@AuthorYear]
- Multiple citations: [@Author1Year; @Author2Year]
- Citation with page number: [@AuthorYear, p. 123]

The add-in will automatically detect these citations and format them according to your chosen style.

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

## Acknowledgements
Dave Shapiro
https://github.com/daveshap/SparsePrimingRepresentations

## License

Copyright 2025 Auvant Advisory Services

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

### Third-Party Libraries

This project uses several open-source libraries:
- Office.js
- Marked.js
- Citation.js
- Office UI Fabric React

All third-party libraries are used in accordance with their respective licenses.
