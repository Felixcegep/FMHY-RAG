import discord
from discord.ext import commands
from discord import app_commands

from Discordquery_faiss import search

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)




@bot.tree.command(name="ask", description="ask a question")
@app_commands.describe(question="Your question")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()  # Discord sait que tu vas répondre plus tard

    not_formated = search(question)
    await interaction.followup.send(not_formated["answer"])

@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Bot ready as {bot.user}")




bot.run("")
