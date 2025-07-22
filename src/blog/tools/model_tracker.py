#!/usr/bin/env python3
"""
Model Usage Tracker for Blog Crew

This module provides utilities to track and display model usage information
for the CrewAI blog generation system.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from blog.crew import Blog


def get_model_usage_info():
    """
    Get comprehensive model usage information for the blog crew.
    
    Returns:
        dict: Dictionary containing usage information
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize blog crew
    blog_crew = Blog()
    crew = blog_crew.crew()
    
    # Get current usage metrics
    usage_metrics = crew.calculate_usage_metrics()
    
    # Get API configuration
    api_config = {
        'api_base': os.getenv('OPENAI_API_BASE'),
        'api_key_set': bool(os.getenv('OPENAI_API_KEY')),
        'api_key_prefix': os.getenv('OPENAI_API_KEY', '')[:10] + '...' if os.getenv('OPENAI_API_KEY') else 'Not set'
    }
    
    # Get agent configurations
    agents_info = []
    for i, agent in enumerate(crew.agents, 1):
        agent_info = {
            'id': i,
            'role': agent.role,
            'model': getattr(agent.llm, 'model', 'Unknown') if hasattr(agent, 'llm') else 'Default',
            'api_base': getattr(agent.llm, 'api_base', 'Default') if hasattr(agent, 'llm') else 'Default'
        }
        agents_info.append(agent_info)
    
    # Test actual model being used
    actual_model_info = None
    try:
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE')
        )
        
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': 'Hi'}],
            max_tokens=5
        )
        
        actual_model_info = {
            'requested_model': 'gpt-4o-mini',
            'actual_model': response.model,
            'usage': dict(response.usage) if response.usage else None
        }
    except Exception as e:
        actual_model_info = {'error': str(e)}
    
    return {
        'timestamp': datetime.now().isoformat(),
        'crew_usage_metrics': dict(usage_metrics),
        'api_configuration': api_config,
        'agents': agents_info,
        'actual_model_test': actual_model_info,
        'crew_config': {
            'num_agents': len(crew.agents),
            'num_tasks': len(crew.tasks),
            'process': str(crew.process),
            'verbose': crew.verbose
        }
    }


def track_crew_execution(topic="AI LLMs"):
    """
    Run the crew and track usage metrics during execution.
    
    Args:
        topic (str): Topic for the blog post
        
    Returns:
        dict: Usage metrics after execution
    """
    load_dotenv()
    
    print("🚀 Starting Blog Crew Execution with Usage Tracking...")
    print("=" * 60)
    
    # Initialize crew
    blog_crew = Blog()
    crew = blog_crew.crew()
    
    # Get initial usage metrics
    initial_usage = crew.calculate_usage_metrics()
    print(f"📊 Initial Usage: {dict(initial_usage)}")
    
    # Prepare inputs
    inputs = {
        'topic': topic,
        'current_year': str(datetime.now().year)
    }
    
    try:
        print(f"\n🎯 Running crew with topic: '{topic}'")
        result = crew.kickoff(inputs=inputs)
        
        # Get final usage metrics
        final_usage = crew.calculate_usage_metrics()
        
        # Calculate the difference
        usage_diff = {
            'total_tokens': final_usage.total_tokens - initial_usage.total_tokens,
            'prompt_tokens': final_usage.prompt_tokens - initial_usage.prompt_tokens,
            'completion_tokens': final_usage.completion_tokens - initial_usage.completion_tokens,
            'successful_requests': final_usage.successful_requests - initial_usage.successful_requests
        }
        
        print(f"\n✅ Execution Complete!")
        print(f"📊 Final Usage: {dict(final_usage)}")
        print(f"📈 Usage for this run: {usage_diff}")
        
        return {
            'initial': dict(initial_usage),
            'final': dict(final_usage),
            'difference': usage_diff,
            'result': str(result) if result else "No result"
        }
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return {'error': str(e)}


def print_usage_summary():
    """Print a formatted summary of model usage information."""
    
    info = get_model_usage_info()
    
    print("=" * 60)
    print("🤖 BLOG CREW MODEL USAGE INFORMATION")
    print("=" * 60)
    
    print(f"\n📅 Report Generated: {info['timestamp']}")
    
    print(f"\n🔧 API Configuration:")
    print(f"   • API Base: {info['api_configuration']['api_base']}")
    print(f"   • API Key: {info['api_configuration']['api_key_prefix']}")
    
    print(f"\n👥 Crew Configuration:")
    print(f"   • Agents: {info['crew_config']['num_agents']}")
    print(f"   • Tasks: {info['crew_config']['num_tasks']}")
    print(f"   • Process: {info['crew_config']['process']}")
    print(f"   • Verbose: {info['crew_config']['verbose']}")
    
    print(f"\n🧠 Agent Models:")
    for agent in info['agents']:
        print(f"   • {agent['role']}: {agent['model']}")
    
    print(f"\n📊 Current Usage Metrics:")
    metrics = info['crew_usage_metrics']
    print(f"   • Total Tokens: {metrics['total_tokens']}")
    print(f"   • Prompt Tokens: {metrics['prompt_tokens']}")
    print(f"   • Completion Tokens: {metrics['completion_tokens']}")
    print(f"   • Successful Requests: {metrics['successful_requests']}")
    
    if 'error' not in info['actual_model_test']:
        print(f"\n🧪 Live Model Test:")
        test = info['actual_model_test']
        print(f"   • Requested: {test['requested_model']}")
        print(f"   • Actually Used: {test['actual_model']}")
        if test['usage']:
            print(f"   • Test Usage: {test['usage']['total_tokens']} tokens")
            if 'cost' in test['usage']:
                print(f"   • Test Cost: ${test['usage']['cost']['total_cost']}")
    else:
        print(f"\n❌ Model Test Error: {info['actual_model_test']['error']}")
    
    print("\n💡 Usage Tips:")
    print("   • Run track_crew_execution() to see live usage during blog generation")
    print("   • Your requests are routed through DynaRoute for optimal model selection")
    print("   • Current routing: gpt-4o-mini → gcp-gemini-2.0-flash-lite")
    print("=" * 60)


if __name__ == "__main__":
    print_usage_summary()
