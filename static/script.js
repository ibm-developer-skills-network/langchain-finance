$(document).ready(function() {
    // Function to send the selected stock ticker to the server
    function sendMessage() {
        var selectedTicker = $("#stockTickerSelect").val();

        // Show the user's message in the chat window
        appendUserMessage(selectedTicker);

        // Show a loading message while waiting for the response
        appendBotMessage("Processing...");

        $.ajax({
            url: '/process-message',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ userMessage: selectedTicker }),
            success: function(response) {
                var botResponse = response.botResponse;
                // Convert newline characters to <br> elements for proper display
                var formattedResponse = botResponse.replace(/\n/g, '<br>');
                // Replace the last bot message with the bot's response using .html() instead of .text()
                $("#chatMessages .bot-message").last().html(formattedResponse);
            },            
            error: function(error) {
                console.error(error);
                appendBotMessage("Error processing request.");
            }
        });
    }

    // Function to append a bot message to the chat area
    function appendBotMessage(message, isInitial = false) {
        var chatMessages = $("#chatMessages");
        var botMessage = $("<div>").addClass("bot-message").text(message);
        if (isInitial) {
            botMessage.addClass("initial-message");
        }
        chatMessages.append(botMessage);
        chatMessages.scrollTop(chatMessages[0].scrollHeight);
    }

    // Function to append a user message to the chat area
    function appendUserMessage(message) {
        var chatMessages = $("#chatMessages");
        var userMessage = $("<div>").addClass("user-message").text(message);
        chatMessages.append(userMessage);
        chatMessages.scrollTop(chatMessages[0].scrollHeight);
    }

    // Bind the submit button click event to send the message
    $("#submitButton").click(sendMessage);

    // Bind the reset button click event to reset the conversation
    $("#resetButton").click(function() {
        $("#chatMessages").empty(); // Clear all messages
        appendBotMessage("Please select a stock ticker to get started.", true);
    });
});
