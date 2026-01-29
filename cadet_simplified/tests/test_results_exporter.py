"""Tests for AnalysisView (results exporter).

Tests the composable view builder for analysis reports including:
- Adding various component types
- Building Panel layouts
- Exporting to HTML
- Download widget generation
"""
# requred if run in e.g. debug mode
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[2]))


import pytest
import pandas as pd
import numpy as np

# Panel is required for these tests
pytest.importorskip("panel")
pytest.importorskip("holoviews")

import panel as pn
import holoviews as hv

from cadet_simplified.analysis import AnalysisView, ViewComponent


@pytest.fixture
def empty_view():
    """Create an empty AnalysisView."""
    return AnalysisView()


@pytest.fixture
def titled_view():
    """Create an AnalysisView with metadata."""
    return AnalysisView(
        title="Test Report",
        author="Test Author",
        description="Test description",
    )


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "time": np.linspace(0, 100, 50),
        "Salt": np.sin(np.linspace(0, 10, 50)) * 50 + 100,
        "Protein": np.exp(-np.linspace(0, 10, 50)) * 10,
    })


@pytest.fixture
def sample_plot():
    """Create a sample HoloViews plot for testing."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    return hv.Curve((x, y), label="Test Curve")



class TestAnalysisViewInit:
    """Tests for AnalysisView initialization."""
    
    def test_empty_init(self, empty_view):
        """Test empty initialization."""
        assert empty_view.title is None
        assert empty_view.author is None
        assert empty_view.description is None
        assert empty_view.is_empty
    
    def test_init_with_metadata(self, titled_view):
        """Test initialization with metadata."""
        assert titled_view.title == "Test Report"
        assert titled_view.author == "Test Author"
        assert titled_view.description == "Test description"
    
    def test_init_sets_created_at(self, empty_view):
        """Test that created_at is set."""
        assert empty_view.created_at is not None



class TestAddComponents:
    """Tests for adding various components to the view."""
    
    def test_add_title(self, empty_view):
        """Test adding a title."""
        result = empty_view.add_title("My Title", level=1)
        
        assert result is empty_view  # Method chaining
        assert len(empty_view) == 1
        assert not empty_view.is_empty
    
    def test_add_title_with_level(self, empty_view):
        """Test adding titles with different levels."""
        empty_view.add_title("H1", level=1)
        empty_view.add_title("H2", level=2)
        empty_view.add_title("H3", level=3)
        
        assert len(empty_view) == 3
        assert empty_view._components[0].options["level"] == 1
        assert empty_view._components[1].options["level"] == 2
        assert empty_view._components[2].options["level"] == 3
    
    def test_add_section(self, empty_view):
        """Test adding a section."""
        result = empty_view.add_section("Section Title")
        
        assert result is empty_view
        assert len(empty_view) == 1
        assert empty_view._components[0].type == "section"
    
    def test_add_text(self, empty_view):
        """Test adding markdown text."""
        result = empty_view.add_text("Some **bold** text")
        
        assert result is empty_view
        assert len(empty_view) == 1
        assert empty_view._components[0].type == "markdown"
        assert empty_view._components[0].data == "Some **bold** text"
    
    def test_add_plot(self, empty_view, sample_plot):
        """Test adding a plot."""
        result = empty_view.add_plot(sample_plot, height=400)
        
        assert result is empty_view
        assert len(empty_view) == 1
        assert empty_view._components[0].type == "plot"
        assert empty_view._components[0].options["height"] == 400
    
    def test_add_table(self, empty_view, sample_dataframe):
        """Test adding a table."""
        result = empty_view.add_table(sample_dataframe, height=200)
        
        assert result is empty_view
        assert len(empty_view) == 1
        assert empty_view._components[0].type == "table"
        assert empty_view._components[0].options["height"] == 200
    
    def test_add_divider(self, empty_view):
        """Test adding a divider."""
        result = empty_view.add_divider()
        
        assert result is empty_view
        assert len(empty_view) == 1
        assert empty_view._components[0].type == "divider"
    
    def test_add_spacer(self, empty_view):
        """Test adding a spacer."""
        result = empty_view.add_spacer(height=30)
        
        assert result is empty_view
        assert len(empty_view) == 1
        assert empty_view._components[0].type == "spacer"
        assert empty_view._components[0].options["height"] == 30
    
    def test_add_alert(self, empty_view):
        """Test adding an alert."""
        result = empty_view.add_alert("Warning message", alert_type="warning")
        
        assert result is empty_view
        assert len(empty_view) == 1
        assert empty_view._components[0].type == "alert"
        assert empty_view._components[0].options["alert_type"] == "warning"
    
    def test_add_raw_html(self, empty_view):
        """Test adding raw HTML."""
        result = empty_view.add_raw_html("<div>Custom HTML</div>")
        
        assert result is empty_view
        assert len(empty_view) == 1
        assert empty_view._components[0].type == "raw_html"



class TestMethodChaining:
    """Tests for fluent method chaining."""
    
    def test_chain_multiple_adds(self, empty_view, sample_dataframe):
        """Test chaining multiple add methods."""
        result = (
            empty_view
            .add_title("Report")
            .add_section("Introduction")
            .add_text("Some text")
            .add_divider()
            .add_table(sample_dataframe)
        )
        
        assert result is empty_view
        assert len(empty_view) == 5
    
    def test_clear_returns_self(self, empty_view):
        """Test that clear returns self."""
        empty_view.add_title("Test")
        result = empty_view.clear()
        
        assert result is empty_view
        assert empty_view.is_empty



class TestViewBuilding:
    """Tests for building Panel layouts."""
    
    def test_view_returns_column(self, empty_view):
        """Test that view() returns a Panel Column."""
        empty_view.add_title("Test")
        result = empty_view.view()
        
        assert isinstance(result, pn.Column)
    
    def test_empty_view_shows_message(self, empty_view):
        """Test that empty view shows a message."""
        result = empty_view.view()
        
        assert isinstance(result, pn.Column)
        # Should contain some indication it's empty
    
    def test_view_contains_all_components(self, empty_view, sample_dataframe):
        """Test that view contains all added components."""
        empty_view.add_title("Title")
        empty_view.add_text("Text")
        empty_view.add_table(sample_dataframe)
        
        result = empty_view.view()
        
        # Column should have multiple objects
        assert len(result.objects) == 3



class TestHtmlExport:
    """Tests for HTML export functionality."""
    
    def test_to_html_returns_string(self, titled_view):
        """Test that to_html returns a string."""
        titled_view.add_title("Test")
        html = titled_view.to_html()
        
        assert isinstance(html, str)
        assert len(html) > 0
    
    def test_to_html_includes_doctype(self, titled_view):
        """Test that HTML includes DOCTYPE."""
        titled_view.add_title("Test")
        html = titled_view.to_html()
        
        assert "<!DOCTYPE html>" in html
    
    def test_to_html_includes_title(self, titled_view):
        """Test that HTML includes the title."""
        titled_view.add_title("Test")
        html = titled_view.to_html()
        
        assert "<title>Test Report</title>" in html
        assert "Test Report" in html
    
    def test_to_html_includes_author(self, titled_view):
        """Test that HTML includes the author."""
        titled_view.add_title("Test")
        html = titled_view.to_html()
        
        assert "Test Author" in html
    
    def test_to_html_includes_bootstrap(self, titled_view):
        """Test that HTML includes Bootstrap CSS."""
        titled_view.add_title("Test")
        html = titled_view.to_html()
        
        assert "bootstrap" in html.lower()
    
    def test_to_html_includes_bokeh_resources(self, titled_view, sample_plot):
        """Test that HTML includes Bokeh resources when plot is added."""
        titled_view.add_plot(sample_plot)
        html = titled_view.to_html()
        
        assert "bokeh" in html.lower()
    
    def test_to_html_converts_markdown_text(self, empty_view):
        """Test that markdown text is converted to HTML."""
        empty_view.add_text("**Bold text**")
        html = empty_view.to_html()
        
        assert "<strong>Bold text</strong>" in html
    
    def test_to_html_converts_titles(self, empty_view):
        """Test that titles are converted to headings."""
        empty_view.add_title("Heading 1", level=1)
        empty_view.add_title("Heading 2", level=2)
        html = empty_view.to_html()
        
        assert "<h1>Heading 1</h1>" in html
        assert "<h2>Heading 2</h2>" in html
    
    def test_to_html_converts_tables(self, empty_view, sample_dataframe):
        """Test that tables are converted to HTML tables."""
        empty_view.add_table(sample_dataframe)
        html = empty_view.to_html()
        
        assert "<table" in html
        assert "time" in html  # Column name
    
    def test_to_html_converts_dividers(self, empty_view):
        """Test that dividers are converted to hr tags."""
        empty_view.add_divider()
        html = empty_view.to_html()
        
        assert "<hr" in html
    
    def test_to_html_converts_alerts(self, empty_view):
        """Test that alerts are converted with Bootstrap classes."""
        empty_view.add_alert("Info message", alert_type="info")
        empty_view.add_alert("Warning message", alert_type="warning")
        html = empty_view.to_html()
        
        assert "alert-info" in html
        assert "alert-warning" in html


class TestDownloadWidget:
    """Tests for download widget generation."""
    
    def test_download_widget_returns_file_download(self, titled_view):
        """Test that download_widget returns a FileDownload widget."""
        titled_view.add_title("Test")
        widget = titled_view.download_widget()
        
        assert isinstance(widget, pn.widgets.FileDownload)
    
    def test_download_widget_custom_filename(self, titled_view):
        """Test download widget with custom filename."""
        titled_view.add_title("Test")
        widget = titled_view.download_widget(filename="custom_report.html")
        
        assert widget.filename == "custom_report.html"
    
    def test_download_widget_custom_label(self, titled_view):
        """Test download widget with custom label."""
        titled_view.add_title("Test")
        widget = titled_view.download_widget(label="Download PDF")
        
        assert widget.label == "Download PDF"
    
    def test_download_widget_button_type(self, titled_view):
        """Test download widget with custom button type."""
        titled_view.add_title("Test")
        widget = titled_view.download_widget(button_type="success")
        
        assert widget.button_type == "success"



class TestViewComponent:
    """Tests for ViewComponent dataclass."""
    
    def test_view_component_creation(self):
        """Test creating a ViewComponent."""
        comp = ViewComponent(
            type="markdown",
            data="Test text",
            options={"key": "value"},
        )
        
        assert comp.type == "markdown"
        assert comp.data == "Test text"
        assert comp.options == {"key": "value"}
    
    def test_view_component_default_options(self):
        """Test ViewComponent default options."""
        comp = ViewComponent(type="divider", data=None)
        
        assert comp.options == {}

class TestMarkdownConversion:
    """Tests for markdown to HTML conversion."""
    
    def test_convert_bold(self, empty_view):
        """Test converting bold text."""
        html = empty_view._markdown_to_html("**bold**")
        assert "<strong>bold</strong>" in html
    
    def test_convert_italic(self, empty_view):
        """Test converting italic text."""
        html = empty_view._markdown_to_html("*italic*")
        assert "<em>italic</em>" in html
    
    def test_convert_inline_code(self, empty_view):
        """Test converting inline code."""
        html = empty_view._markdown_to_html("`code`")
        assert "<code>code</code>" in html
    
    def test_convert_links(self, empty_view):
        """Test converting links."""
        html = empty_view._markdown_to_html("[link](http://example.com)")
        assert '<a href="http://example.com">link</a>' in html
    
    def test_convert_headers(self, empty_view):
        """Test converting headers."""
        html = empty_view._markdown_to_html("# Header 1")
        assert "<h1>Header 1</h1>" in html
        
        html = empty_view._markdown_to_html("## Header 2")
        assert "<h2>Header 2</h2>" in html
    
    def test_convert_unordered_list(self, empty_view):
        """Test converting unordered list."""
        md = "- Item 1\n- Item 2"
        html = empty_view._markdown_to_html(md)
        
        assert "<ul>" in html
        assert "<li>Item 1</li>" in html
        assert "<li>Item 2</li>" in html
        assert "</ul>" in html


class TestCompleteReport:
    """Tests for building complete reports."""
    
    def test_build_chromatography_report(self, sample_dataframe, sample_plot):
        """Test building a typical chromatography report."""
        view = AnalysisView(
            title="Chromatography Analysis",
            author="Lab Team",
            description="Analysis of gradient elution experiments",
        )
        
        view.add_title("Results Overview", level=2)
        view.add_text("This report summarizes the chromatography experiments.")
        
        view.add_section("Chromatogram")
        view.add_plot(sample_plot, height=400)
        
        view.add_section("Data Summary")
        view.add_table(sample_dataframe, height=200)
        
        view.add_spacer(height=20)
        view.add_alert("All experiments completed successfully", alert_type="success")
        
        # Test that view builds
        panel_view = view.view()
        assert isinstance(panel_view, pn.Column)
        
        # Test that HTML exports
        html = view.to_html()
        assert "Chromatography Analysis" in html
        assert "Results Overview" in html
        assert "Data Summary" in html
    
    def test_report_with_many_sections(self):
        """Test report with many sections doesn't break."""
        view = AnalysisView(title="Large Report")
        
        for i in range(20):
            view.add_section(f"Section {i}")
            view.add_text(f"Content for section {i}")
        
        assert len(view) == 40  # 20 sections + 20 texts
        
        html = view.to_html()
        assert "Section 19" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
