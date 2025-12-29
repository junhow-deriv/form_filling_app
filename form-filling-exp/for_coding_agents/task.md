# Task

You are a world-class coding expert responsible for building a form filling app. There is already an initial scaffold of an app in the current repo. Your goal is to extend the app to meet the requirements.

## Specifications

The specs are as follows, mostly focused on what to extend the app towards.

### Overall User Flow
- I want to make the app a nextjs web app that can (eventually) be hosted on Vercel. Don't worry about that last part, just make this a bit more complete as an app.
- The app should be a multi-turn chat interface on the right hand side, and a form upload option on the left hand side. 
- The user uploads a blank form in the left hand side. We will then surface the detected fields. 
- The user then types an initial chat message in the right hand side. The underlying form filling agent will then try to fill out the form.
- It will return the rendered form on the left hand side (including a toggle so the user can switch between the blank form and the filled form). 
- The user can then decide to continue the chat conversation in order to modify the filled values, if needed.
- The user can download the filled form at any point during the chat. 
- (Extension, P2, don't do yet) The user should be able to directly modify the form values through the web app 
- (Extension, P3, don't do yet) Besides the user text prompts, the user should be able to upload file attachments. We would want to parse it through LlamaParse on the backend, and attach the text context into the prompt window (or store as parsed files that the agent can traverse).

### Backend
- There is existing code to detect the fillable form fields. That code should largely be ok, modify the interfaces if needed to adapt to the app.
- There is an existing form filling agent powered by the Anthropic LLM API as well as the agent SDK API.
- I want to delete the no AI option, and leave the agent option as the default one. You don't need to remove the single-shot option and the code, but mark it as deprecated or legacy. 
- The current agent implementation works fine. Modify it as needed to conform to the UI/UX, but the overall agent loop + tools should largely be correct.
- There is no concept of a user login right now, so if the user refreshes the page or opens a new tab, then the app will reset. You could try to link to a given session through URL. we can worry about auth and stuff later. 


### Frontend / UI 
- Make it look aesthetically similar to other products we have in LlamaCloud. This includes our parse feature as well as our coding agent endpoint for building agent workflows (see llamacloud_parse.png and llamacloud_agent_builder.png). Though obviously make sure you include the features that conform to the specs above. 



## Context
All initial context is in `for_coding_agents`. Besides reading from current files, you can also use it as a scratch space to store files for later.
- Initial context for the app is in initial_cursor_chat.md.
- There are also images llamacloud_parse.png and llamacloud_agent_builder.png. 

## Additional Notes

### Overall Behavior
- Make sure the code is maintainable, clean, and not overly verbose
- You are a staff engineer so don't make stupid decisions 