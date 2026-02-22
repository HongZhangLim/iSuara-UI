package com.isuara.app.service

import android.util.Log
import com.google.ai.client.generativeai.GenerativeModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * GeminiTranslator — translates detected BIM sign keywords into
 * a natural Bahasa Melayu sentence using Gemini 2.5 Flash Lite.
 */
class GeminiTranslator(apiKey: String) {

    companion object {
        private const val TAG = "GeminiTranslator"
    }

    private val model = GenerativeModel(
        modelName = "gemini-2.0-flash-lite",
        apiKey = apiKey
    )

    /**
     * Convert a list of BIM sign keywords into a natural sentence.
     * Runs on IO dispatcher. Returns the translated sentence or error string.
     */
    suspend fun translate(words: List<String>): String = withContext(Dispatchers.IO) {
        if (words.isEmpty()) return@withContext ""

        val prompt = """
You are a professional Malaysian BIM sign language interpreter.

Rules:
- Input is a list of keywords from sign language.
- Rearrange into a natural Bahasa Melayu sentence which follow the format subject+verb+object+time.
- Add missing pronouns or grammar if needed.
- Keep meaning accurate.
- Do NOT explain anything.
- Return only one sentence.

Input: $words
Output:
""".trimIndent()

        try {
            val response = model.generateContent(prompt)
            val text = response.text?.trim() ?: ""
            Log.i(TAG, "Translation: $words → $text")
            text
        } catch (e: Exception) {
            Log.e(TAG, "Translation failed", e)
            "[Translation error: ${e.message}]"
        }
    }
}
