"""
REST API User Data Adapter
Connects DKI to external REST APIs for user data

Author: AGI Demo Project
Version: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from dki.adapters.base import (
    IUserDataAdapter,
    AdapterConfig,
    AdapterType,
    UserProfile,
    ChatMessage,
    UserPreference,
)


class RESTAPIUserDataAdapter(IUserDataAdapter):
    """
    REST API adapter for user data.
    
    Connects to external REST APIs to fetch user data.
    Supports both aiohttp (async) and requests (sync) HTTP clients.
    
    Expected API Endpoints:
    - GET /users/{user_id} -> UserProfile
    - GET /sessions/{session_id}/messages -> List[ChatMessage]
    - GET /users/{user_id}/preferences -> List[UserPreference]
    - GET /users/{user_id}/search?q={query} -> List[ChatMessage]
    
    Example:
        config = AdapterConfig(
            adapter_type=AdapterType.REST_API,
            base_url="https://api.example.com/v1",
            api_key="your-api-key",
        )
        
        async with RESTAPIUserDataAdapter(config) as adapter:
            profile = await adapter.get_user_profile("user_123")
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        """Initialize REST API adapter."""
        super().__init__(config)
        
        if self.config.adapter_type != AdapterType.REST_API:
            self.config.adapter_type = AdapterType.REST_API
        
        self._session = None
        self._use_aiohttp = True
        
        # Endpoint paths (can be customized via config.options)
        self._endpoints = {
            "user_profile": "/users/{user_id}",
            "session_messages": "/sessions/{session_id}/messages",
            "user_preferences": "/users/{user_id}/preferences",
            "search_history": "/users/{user_id}/search",
            "health": "/health",
        }
        
        # Override endpoints from config
        if self.config.options.get("endpoints"):
            self._endpoints.update(self.config.options["endpoints"])
    
    async def connect(self) -> None:
        """Initialize HTTP client session."""
        if self._connected:
            return
        
        try:
            import aiohttp
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            # Add custom headers from config
            if self.config.options.get("headers"):
                headers.update(self.config.options["headers"])
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(
                base_url=self.config.base_url,
                headers=headers,
                timeout=timeout,
            )
            
            self._connected = True
            self._use_aiohttp = True
            logger.info(f"Connected to REST API (aiohttp): {self.config.base_url}")
            
        except ImportError:
            logger.warning("aiohttp not installed, falling back to requests")
            self._use_aiohttp = False
            
            try:
                import requests
                
                self._session = requests.Session()
                self._session.headers.update({
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                })
                
                if self.config.api_key:
                    self._session.headers["Authorization"] = f"Bearer {self.config.api_key}"
                
                if self.config.options.get("headers"):
                    self._session.headers.update(self.config.options["headers"])
                
                self._connected = True
                logger.info(f"Connected to REST API (requests): {self.config.base_url}")
                
            except Exception as e:
                logger.error(f"Failed to initialize HTTP client: {e}")
                raise
    
    async def disconnect(self) -> None:
        """Close HTTP client session."""
        if not self._connected:
            return
        
        try:
            if self._use_aiohttp and self._session:
                await self._session.close()
            elif self._session:
                self._session.close()
            
            self._connected = False
            self._session = None
            logger.info("Disconnected from REST API")
            
        except Exception as e:
            logger.error(f"Error disconnecting from REST API: {e}")
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request to API."""
        if not self._connected:
            await self.connect()
        
        url = endpoint if endpoint.startswith("http") else endpoint
        
        try:
            if self._use_aiohttp:
                async with self._session.request(
                    method,
                    url,
                    params=params,
                    json=json_data,
                ) as response:
                    if response.status == 404:
                        return None
                    response.raise_for_status()
                    return await response.json()
            else:
                full_url = f"{self.config.base_url}{url}"
                response = self._session.request(
                    method,
                    full_url,
                    params=params,
                    json=json_data,
                    timeout=self.config.timeout,
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"API request failed: {method} {url} - {e}")
            return None
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from REST API."""
        endpoint = self._endpoints["user_profile"].format(user_id=user_id)
        
        data = await self._request("GET", endpoint)
        
        if data:
            try:
                # Handle nested data structure
                if "data" in data:
                    data = data["data"]
                
                # Ensure user_id is set
                data["user_id"] = data.get("user_id") or data.get("id") or user_id
                
                return UserProfile.from_dict(data)
            except Exception as e:
                logger.error(f"Error parsing user profile: {e}")
        
        return None
    
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> List[ChatMessage]:
        """Get session conversation history from REST API."""
        endpoint = self._endpoints["session_messages"].format(session_id=session_id)
        
        params = {"limit": limit}
        if before:
            params["before"] = before.isoformat()
        if after:
            params["after"] = after.isoformat()
        
        data = await self._request("GET", endpoint, params=params)
        
        if data:
            try:
                # Handle nested data structure
                messages_data = data.get("data") or data.get("messages") or data
                
                if isinstance(messages_data, list):
                    messages = []
                    for msg_data in messages_data:
                        # Ensure required fields
                        msg_data["message_id"] = msg_data.get("message_id") or msg_data.get("id")
                        msg_data["session_id"] = msg_data.get("session_id") or session_id
                        
                        messages.append(ChatMessage.from_dict(msg_data))
                    return messages
                    
            except Exception as e:
                logger.error(f"Error parsing session history: {e}")
        
        return []
    
    async def get_user_preferences(
        self,
        user_id: str,
        preference_types: Optional[List[str]] = None,
        include_expired: bool = False,
    ) -> List[UserPreference]:
        """Get user preferences from REST API."""
        endpoint = self._endpoints["user_preferences"].format(user_id=user_id)
        
        params = {}
        if preference_types:
            params["types"] = ",".join(preference_types)
        if include_expired:
            params["include_expired"] = "true"
        
        data = await self._request("GET", endpoint, params=params)
        
        if data:
            try:
                # Handle nested data structure
                prefs_data = data.get("data") or data.get("preferences") or data
                
                if isinstance(prefs_data, list):
                    preferences = []
                    for pref_data in prefs_data:
                        # Ensure required fields
                        pref_data["user_id"] = pref_data.get("user_id") or user_id
                        
                        preferences.append(UserPreference.from_dict(pref_data))
                    
                    # Sort by priority
                    preferences.sort(key=lambda p: p.priority, reverse=True)
                    return preferences
                    
            except Exception as e:
                logger.error(f"Error parsing user preferences: {e}")
        
        return []
    
    async def search_relevant_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """Search for relevant messages via REST API."""
        endpoint = self._endpoints["search_history"].format(user_id=user_id)
        
        params = {
            "q": query,
            "limit": limit,
        }
        if session_id:
            params["session_id"] = session_id
        
        data = await self._request("GET", endpoint, params=params)
        
        if data:
            try:
                # Handle nested data structure
                messages_data = data.get("data") or data.get("results") or data.get("messages") or data
                
                if isinstance(messages_data, list):
                    messages = []
                    for msg_data in messages_data:
                        msg_data["message_id"] = msg_data.get("message_id") or msg_data.get("id")
                        messages.append(ChatMessage.from_dict(msg_data))
                    return messages
                    
            except Exception as e:
                logger.error(f"Error parsing search results: {e}")
        
        return []
    
    async def health_check(self) -> bool:
        """Check REST API health."""
        endpoint = self._endpoints["health"]
        
        try:
            data = await self._request("GET", endpoint)
            
            if data:
                status = data.get("status") or data.get("health")
                return status in ("ok", "healthy", "up", True)
            
            return False
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def save_message(self, message: ChatMessage) -> bool:
        """Save a message via REST API."""
        endpoint = self._endpoints["session_messages"].format(
            session_id=message.session_id
        )
        
        try:
            data = await self._request(
                "POST",
                endpoint,
                json_data=message.to_dict(),
            )
            return data is not None
            
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return False
    
    async def update_user_preference(self, preference: UserPreference) -> bool:
        """Update or create a user preference via REST API."""
        endpoint = self._endpoints["user_preferences"].format(
            user_id=preference.user_id
        )
        
        try:
            if preference.preference_id:
                # Update existing
                endpoint = f"{endpoint}/{preference.preference_id}"
                data = await self._request(
                    "PUT",
                    endpoint,
                    json_data=preference.to_dict(),
                )
            else:
                # Create new
                data = await self._request(
                    "POST",
                    endpoint,
                    json_data=preference.to_dict(),
                )
            
            return data is not None
            
        except Exception as e:
            logger.error(f"Error updating preference: {e}")
            return False
