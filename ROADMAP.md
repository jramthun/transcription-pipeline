# Future Direction(s)
The following features/options may or may not be added, but are ideas that I believe complement and/or improve the functionality of this project.

### Automated Speaker Labeling and NLP tasks
- Using something like [SpaCy](https://spacy.io) to conduct: Named Entity Recognition, coreference resolution, etc. to identify and map names present in transcript to generated speaker labels (e.g., mapping speaker_2 -> Sen. Wyden w/o human input)
- Utilize information included in the hearing page to help identify members present and witness names (can be used to correct the transcript by providing a custom vocabulary for this inference)
- Utilize known information about each committee to provide a custom vocabulary for the names of members who may be and/or are present
    - Specifically to avoid training model(s) capable of identifying 538 people every two years

### RAG-based interfacing and other related utilities
- From my brief experiences in the policy world, every company/customer has their own format for summaries and briefs. Having seen a handful of companies produce samples of hearing summaries, I've only seen them appear in the producer's format (which aren't as customizable). Additionally, summaries are somewhat limited in their utility. So, I figured it would be beneficial to instead create a resource that can produce the information as desired by a user: a 'chat' model familiar with the hearing
- This would at least include features like summarization and DocQA (more TBD), but could allow someone to quickly determine if their keyword(s) appeared in the hearing and in what context
- This could potentially grow (depending on implementation) to allow for cross-hearing knowledge

### Subtitles for RLHF
- The interim steps outputs all the necessary information to construct complete subtitles that can be added to the original video
- This enables us to collect feedback from human reviewers to improve model and pipeline performance

### Fine-tuning using domain-specific data
As Congressional hearings are part of public domain, they make for a convenient, license free source of thousands of hours of multi-speaker audio and video data. The issue? It's not labeled. However, with tools like this project, it may become possible to process this large repository of data using published transcripts from the Library of Congress. Using this, we could not only use this data to improve the accuracy and precision of the models involved, but we could also present this as a larger open dataset for others to develop new models. Sounds like a win-win.

### Web interfacing & Deployment
I hope to deploy this project one day and it would be much more convenient for non-technical users to be able to rely on something they are familiar with: a web browser. As such, we can develop a web-based interface to enable extra functionaity for these end users, including:
- Simple hearing selection based on the Congressional calendar
- Requesting and/or scheduling "coverage" of upcoming hearings based on the Congressional calendar
- Progress indication (as processing long-form audio takes a long time) or notification (e.g., emailing a user once 'their' hearing is ready)
- Storing/caching previously completed hearings (since multiple users may want the same hearing)
- LLM-based interaction with the content to allow for custom information requests (requires [RAG-based interfacing and other related utilities](#rag-based-interfacing-and-other-related-utilities) to be accomplished)

Additionally, it's important to develop something like this in a scalable and secure manner, so it would be smart to containerize each core functionality to handle "jobs". This divided approach would allow for "jobs" to be scheduled in a way that prevents a user from experiencing delays.
- Each core function would be its own container, allowing for jobs to be completed in end-to-end fashion and minimizing repeat work (e.g., caching previously completed jobs unless error-filled)
- Jobs can be scheduled/queued, allowing for optimization of delivery
- Work is no longer limited to a monolithic structure, allowing for distribution of work across multiple machines

Of course, this adds significant complexity but comes with many potential benefits.