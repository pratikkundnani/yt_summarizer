import { Hono } from 'hono';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { ChatCloudflareWorkersAI } from '@langchain/cloudflare';
import { stream } from 'hono/streaming';
import { YoutubeLoader } from '@langchain/community/document_loaders/web/youtube';
import { loadQAMapReduceChain, loadSummarizationChain } from 'langchain/chains';
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
		model: '@cf/meta/llama-3-8b-instruct',
		cloudflareAccountId: c.env.CLOUDFLARE_ACCOUNT_ID,
		cloudflareApiToken: c.env.CLOUDFLARE_API_KEY,
		streaming: true,
	});
	// const body = await c.req.json();
	const body = {
		"videoUrl" : "https://youtu.be/qaPMdcCqtWk?si=0UXbTlSXda8cLVSi"
	}
	try {
		const loader = YoutubeLoader.createFromUrl(body.videoUrl);
		const transcript = await loader.load();
		const splitter = new RecursiveCharacterTextSplitter({
			chunkSize: 7000,
			chunkOverlap: 1000
		});
		let docs = await splitter.splitDocuments(transcript);
		// console.log(docs);
		const mapReduceChain = loadQAMapReduceChain(model);

		const res = await mapReduceChain.invoke({
			input_documents: docs,
			question: "You are an expert in summarizing YouTube videos.Your goal is to create a summary of a video from the transcript provided. Provide a detailed bullet point summary."
		});

		//{
		// 			input_documents: docs,
		// 			question: " You are an expert in summarizing YouTube videos.Your goal is to create a summary of a video from the transcript provided. Provide a detailed summary."
		// 		}

		return c.json({res});
	} catch (e) {
		console.log('error', e);
		return c.json({ error: 'An error occurred during processing.' }, 500);
	}
});

export default app;
