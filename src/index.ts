import { Hono } from 'hono';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { ChatCloudflareWorkersAI } from '@langchain/cloudflare';
import { stream } from 'hono/streaming';
import { YoutubeLoader } from '@langchain/community/document_loaders/web/youtube';
import { loadSummarizationChain } from 'langchain/chains';
import { loadQAStuffChain } from 'langchain/chains';
import { PromptTemplate } from '@langchain/core/prompts';

type Bindings = {
	AI: Ai,
	CLOUDFLARE_ACCOUNT_ID: string,
	CLOUDFLARE_API_KEY: string,
	LANGCHAIN_API_KEY: string,
	LANGCHAIN_PROJECT: string,
}

const app = new Hono<{ Bindings: Bindings }>();

app.get('/', async (c) => {
	const model = new ChatCloudflareWorkersAI({
		model: '@hf/thebloke/llama-2-13b-chat-awq',
		cloudflareAccountId: c.env.CLOUDFLARE_ACCOUNT_ID,
		cloudflareApiToken: c.env.CLOUDFLARE_API_KEY,
		streaming: true
	});
	const body = await c.req.json();
	try {
		const loader = YoutubeLoader.createFromUrl(body.videoUrl);
		const transcript = await loader.load();
		const splitter = new RecursiveCharacterTextSplitter({
			chunkSize: 5000,
			chunkOverlap: 400
		});
		let docs = await splitter.splitDocuments(transcript);
		console.log(docs);

		const summaryTemplate = `
You are an expert in summarizing YouTube videos.
Your goal is to create a summary of a youtube video.
Below you find the transcript of a video:
{text};`
		const SUMMARY_PROMPT = PromptTemplate.fromTemplate(summaryTemplate);

		const summaryRefineTemplate = `
You are an expert in summarizing YouTube videos.
Your goal is to create a summary of a video.
We have provided an existing summary up to a certain point: {existing_answer}

Below you find the transcript of a video:
--------
{text}
--------

Given the new context, refine the summary. Provide the summary directly, without any introductory phrases. `;

		const SUMMARY_REFINE_PROMPT = PromptTemplate.fromTemplate(
			summaryRefineTemplate
		);
		const summarizeChain = loadSummarizationChain(model, {
			type: "refine",
			questionPrompt: SUMMARY_PROMPT,
			refinePrompt: SUMMARY_REFINE_PROMPT,
		});

		return stream(c, async (stream) => {
			stream.onAbort(() => {
				console.log('Aborted!');
			});
			const chatstream = await summarizeChain.stream({input_documents: docs});
			const reader = chatstream.getReader();
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				console.log(value.output_text);
				await stream.writeln(value?.output_text.toString());
			}

			console.log('STREAM DONE');
			// stream.pipe(chatStream);
		});
	} catch (e) {
		console.log('error', e);
		return c.json({ error: 'An error occurred during processing.' }, 500);
	}
});

export default app;
