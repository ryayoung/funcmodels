[
    {
        role: 'user',
        content: 'Hi there, how are you?',
    },
    {
        role: 'assistant',
        content: null,
        tool_calls: [
            {
                id: '1',
                type: 'function',
                function: {
                    name: 'get_weather',
                    arguments: '{"city": "Berlin"}',
                }
            },
            {
                id: '2',
                type: 'function',
                function: {
                    name: 'get_stock_price',
                    arguments: '{"ticker": "AAPL"}',
                }
            }
        ]
    },
    {
        role: 'tool',
        content: '20 degrees and sunny',
        tool_call_id: '1',
    },
    {
        role: 'tool',
        content: 'AAPL: $187',
        tool_call_id: '2',
    }
]
