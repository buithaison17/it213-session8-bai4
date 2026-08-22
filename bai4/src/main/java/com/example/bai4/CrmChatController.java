package com.example.bai4;

import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/chat")
@RequiredArgsConstructor
public class CrmChatController {
    private static final String groundingPrompt = """
            Bạn là trợ lý AI thông minh chuyên hỗ trợ tài liệu khách hàng cho hệ thống CRM.
            Nhiệm vụ của bạn là chỉ sử dụng những thông tin được cung cấp trong phần tài liệu ngữ cảnh (Context) để trả lời.
            Quy tắc:
            1. Tuyệt đối không tự suy đoán hoặc bịa đặt thông tin không có trong tài liệu.
            2. Nếu tài liệu không chứa câu trả lời, hãy trả lời chính xác: 'Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu CRM được cung cấp.'
            3. Trả lời súc tích, lịch sự, đúng trọng tâm.
            """;
    private final ChatClient chatClient;
    private final PgVectorStore pgVectorStore;

    @PostMapping
    public String chat(@RequestParam String message) {
        SearchRequest searchRequest = SearchRequest.builder()
                .query(message)
                .topK(3)
                .similarityThreshold(0.3)
                .build();
        return chatClient
                .prompt()
                .system(groundingPrompt)
                .user(message)
                .advisors(QuestionAnswerAdvisor.builder(pgVectorStore).searchRequest(searchRequest).build())
                .call()
                .content();
    }
}
